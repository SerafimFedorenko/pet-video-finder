"""Бизнес-логика анализа видео: детекция, вырезание и оценка кадров."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import cv2
import numpy as np
from PIL import Image

from config import (
    CAT_CLASS_ID,
    COMPARISON_BACKEND,
    DOG_CLASS_ID,
    IMAGE_SIZE,
    MODE_IMAGE,
    MODE_TEXT,
)

ProgressCallback = Callable[[int], None]  # процент выполнения 0..100


class TextPredictor(Protocol):
    def predict_by_text(self, text: str, image: Image.Image) -> float: ...


class ImagePredictor(Protocol):
    def predict_by_img(
        self, reference: Image.Image, image: Image.Image
    ) -> float: ...


@dataclass(frozen=True)
class AnalysisRequest:
    """Параметры одного запуска анализа видео."""

    class_index: int
    interval: float
    mode: str
    description: str = ""
    reference_image: Image.Image | None = None

    def validate(self) -> None:
        if self.mode not in (MODE_TEXT, MODE_IMAGE):
            raise ValueError(f"Неизвестный режим сравнения: {self.mode}")
        if self.interval <= 0:
            raise ValueError("Интервал между кадрами должен быть больше нуля.")
        if self.mode == MODE_TEXT and not self.description.strip():
            raise ValueError("Введите описание для сравнения.")
        if self.mode == MODE_IMAGE and self.reference_image is None:
            raise ValueError("Выберите изображение для сравнения.")


@dataclass(frozen=True)
class FrameResult:
    """Один найденный кадр и результат его сравнения."""

    image: Image.Image
    crops: tuple[Image.Image, ...]
    best_crop_index: int
    score: float
    timestamp: float


@dataclass(frozen=True)
class AnalysisResult:
    frames: tuple[FrameResult, ...] = field(default_factory=tuple)


class VideoAnalysisController:
    """Обрабатывает видео и возвращает готовые результаты для UI.

    Все тяжёлые зависимости (YOLO и модель сравнения) можно подменить
    в тестах через аргументы конструктора.
    """

    def __init__(
        self,
        yolo=None,
        text_predictors: dict[int, TextPredictor] | None = None,
        image_predictors: dict[int, ImagePredictor] | None = None,
    ):
        if yolo is None:
            from recognizer import Recognizer_MP

            yolo = Recognizer_MP()

        if text_predictors is None or image_predictors is None:
            if COMPARISON_BACKEND == "legacy":
                import networks

                if text_predictors is None:
                    text_predictors = {
                        DOG_CLASS_ID: networks.DogsTextPredictor(),
                        CAT_CLASS_ID: networks.CatsTextPredictor(),
                    }
                if image_predictors is None:
                    image_predictors = {
                        DOG_CLASS_ID: networks.DogsImgPredictor(),
                        CAT_CLASS_ID: networks.CatsImgPredictor(),
                    }
            elif COMPARISON_BACKEND == "clip":
                from comparison import create_predictors

                default_text, default_image = create_predictors()
                if text_predictors is None:
                    text_predictors = default_text
                if image_predictors is None:
                    image_predictors = default_image
            else:
                raise ValueError(
                    "Неизвестный backend сравнения: "
                    f"{COMPARISON_BACKEND!r}. Ожидается 'clip' или 'legacy'."
                )

        self.yolo = yolo
        self._text_predictors = dict(text_predictors)
        self._image_predictors = dict(image_predictors)

    # ---------- публичный API ----------

    def estimate_time(self, video_path: str | Path, request: AnalysisRequest) -> float:
        """Оценка времени обработки видео в секундах по одному кадру."""
        request.validate()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise IOError(f"Не удалось открыть видео: {video_path}")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            all_frames_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            ret, frame = capture.read()
            if not ret:
                raise IOError("Видео не содержит читаемых кадров")

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Прогреваем YOLO и модель сравнения: загрузка CLIP и первый
            # инференс не должны попадать в оценку времени обработки.
            self.yolo.predict_img(frame)
            self._score_crops((image,), request)

            start = time.perf_counter()
            self.yolo.predict_img(frame)
            self._score_crops((image,), request)
            elapsed = time.perf_counter() - start
        finally:
            capture.release()

        if all_frames_count <= 0:
            return 0.0
        sampled_frames = all_frames_count / max(1.0, fps * request.interval)
        return sampled_frames * elapsed

    def analyze(
        self,
        video_path: str | Path,
        request: AnalysisRequest,
        progress_cb: ProgressCallback | None = None,
    ) -> AnalysisResult:
        """Находит кадры с животным и оценивает их соответствие запросу."""
        request.validate()

        frames, boxes = self.yolo.predict(
            video_path,
            interval=request.interval,
            index=request.class_index,
            progress_cb=self._yolo_progress(progress_cb),
        )
        if not frames:
            self._report(progress_cb, 100)
            return AnalysisResult()

        total = len(frames)
        frame_results: list[FrameResult] = []
        timings = list(getattr(self.yolo, "timings", []) or [])

        for position, (frame, frame_boxes) in enumerate(zip(frames, boxes)):
            crops = self._crop_frame(frame, frame_boxes)
            score, best_index = self._score_crops(crops, request)
            annotated = self._annotate_frame(frame, frame_boxes, best_index)
            timestamp = timings[position] if position < len(timings) else 0.0
            frame_results.append(
                FrameResult(
                    image=annotated,
                    crops=crops,
                    best_crop_index=best_index,
                    score=score,
                    timestamp=float(timestamp),
                )
            )
            self._report(progress_cb, 50 + int((position + 1) / total * 50))

        self._report(progress_cb, 100)
        return AnalysisResult(tuple(frame_results))

    # ---------- внутренние методы ----------

    @staticmethod
    def _report(progress_cb: ProgressCallback | None, value: int) -> None:
        if progress_cb is not None:
            progress_cb(max(0, min(100, value)))

    def _yolo_progress(self, progress_cb: ProgressCallback | None):
        def callback(processed: int, total: int) -> None:
            if progress_cb is not None and total > 0:
                progress_cb(int(processed / total * 50))

        return callback

    def _predictor(self, request: AnalysisRequest):
        predictors = (
            self._text_predictors
            if request.mode == MODE_TEXT
            else self._image_predictors
        )
        try:
            return predictors[request.class_index]
        except KeyError as exc:
            raise ValueError(
                f"Нет модели для класса животных {request.class_index}."
            ) from exc

    def _score_crops(
        self, crops: Sequence[Image.Image], request: AnalysisRequest
    ) -> tuple[float, int]:
        if not crops:
            return 0.0, 0

        predictor = self._predictor(request)
        crops_list = list(crops)
        if request.mode == MODE_TEXT:
            batch_scorer = getattr(predictor, "predict_many_by_text", None)
            if batch_scorer is not None:
                scores = [float(value) for value in batch_scorer(
                    request.description, crops_list
                )]
            else:
                scores = [
                    float(predictor.predict_by_text(request.description, crop))
                    for crop in crops_list
                ]
        else:
            batch_scorer = getattr(predictor, "predict_many_by_img", None)
            if batch_scorer is not None:
                scores = [float(value) for value in batch_scorer(
                    request.reference_image, crops_list
                )]
            else:
                scores = [
                    float(predictor.predict_by_img(request.reference_image, crop))
                    for crop in crops_list
                ]

        if len(scores) != len(crops_list):
            raise RuntimeError(
                "Модель сравнения вернула неверное число оценок: "
                f"{len(scores)} вместо {len(crops_list)}."
            )

        best_index = max(range(len(scores)), key=scores.__getitem__)
        return scores[best_index], best_index

    @staticmethod
    def _crop_frame(frame: np.ndarray, boxes) -> tuple[Image.Image, ...]:
        height, width = frame.shape[:2]
        crops: list[Image.Image] = []
        for box in boxes:
            x1 = max(0, min(width, int(box[0])))
            y1 = max(0, min(height, int(box[1])))
            x2 = max(0, min(width, int(box[2])))
            y2 = max(0, min(height, int(box[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            crops.append(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)))

        if not crops:
            # Защита от пустых боксов: дальше по коду кадр всегда имеет кроп.
            crops.append(Image.new("RGB", IMAGE_SIZE))
        return tuple(crops)

    @staticmethod
    def _annotate_frame(frame: np.ndarray, boxes, best_index: int) -> Image.Image:
        picture = frame.copy()
        for box_index, box in enumerate(boxes):
            color = (0, 0, 255) if box_index == best_index else (0, 255, 0)
            cv2.rectangle(
                picture,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color,
                4,
            )
        return Image.fromarray(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
