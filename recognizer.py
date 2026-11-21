"""Распознавание животных на видео с помощью YOLO."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
from ultralytics import YOLO

from config import CAT_CLASS_ID, DOG_CLASS_ID, PROJECT_ROOT

# Индексы классов COCO, которые использует приложение.
__all__ = ["Recognizer", "Recognizer_MP", "CAT_CLASS_ID", "DOG_CLASS_ID"]

ProgressCallback = Callable[[int, int], None]


class Recognizer:
    """Последовательная обработка видео."""

    def __init__(self, model_path: str | Path | None = None, confidence: float = 0.6):
        self.model_path = Path(model_path) if model_path else PROJECT_ROOT / "yolov8n.pt"
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Не найден файл модели YOLO: {self.model_path}. "
                "Скачайте yolov8n.pt в каталог проекта."
            )
        # Второй аргумент ("v8") в новых версиях ultralytics не нужен:
        # задача определяется автоматически по файлу весов.
        self.model = YOLO(str(self.model_path))
        self.confidence = confidence
        self.timings: list[float] = []

    def predict_img(self, img_cv2, confidence: float | None = None) -> np.ndarray:
        """Возвращает массив боксов (x1, y1, x2, y2, conf, class_id)."""
        if img_cv2 is None:
            return np.empty((0, 6))
        results = self.model(
            img_cv2,
            save=False,
            conf=self.confidence if confidence is None else confidence,
            verbose=False,
        )
        if results and results[0].boxes is not None:
            return results[0].boxes.data.cpu().numpy()
        return np.empty((0, 6))

    @staticmethod
    def check_index(results, index: int):
        """Оставляет только объекты нужного класса."""
        return [result for result in results if int(result[5]) == index]

    @staticmethod
    def _open_video(video_path: str | Path) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise IOError(f"Не удалось открыть видео: {video_path}")
        return capture

    def _iter_frames(
        self, video_path: str | Path, interval: float
    ) -> Iterator[tuple[int, np.ndarray, float, int]]:
        """Перебирает кадры с заданным интервалом (в секундах).

        Возвращает (номер кадра, кадр, fps, всего кадров для обработки).
        """
        capture = self._open_video(video_path)
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(1, int(round(fps * max(interval, 0.001))))
            total_samples = (
                int(np.ceil(total_frames / step)) if total_frames > 0 else 0
            )
            frame_index = 0
            while total_frames <= 0 or frame_index < total_frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = capture.read()
                if not ret:
                    break
                yield frame_index, frame, fps, total_samples
                frame_index += step
        finally:
            capture.release()

    def predict(
        self,
        video_path: str | Path,
        interval: float = 1,
        index: int = DOG_CLASS_ID,
        progress_cb: ProgressCallback | None = None,
        batch_size: int = 1,
    ) -> tuple[list[np.ndarray], list[list]]:
        """Ищет кадры, на которых есть объект класса index.

        Возвращает список кадров и список найденных объектов для каждого кадра.
        """
        self.timings = []
        frames_pets: list[np.ndarray] = []
        results_pets: list[list] = []

        batch_frames: list[np.ndarray] = []
        batch_times: list[float] = []
        total_samples = 0
        processed = 0

        def flush() -> None:
            nonlocal processed
            if not batch_frames:
                return
            batch_results = self.model(
                batch_frames,
                save=False,
                conf=self.confidence,
                verbose=False,
            )
            for frame, timestamp, result in zip(
                batch_frames, batch_times, batch_results
            ):
                boxes = (
                    result.boxes.data.cpu().numpy()
                    if result.boxes is not None
                    else np.empty((0, 6))
                )
                needed = self.check_index(boxes, index)
                processed += 1
                if needed:
                    frames_pets.append(frame)
                    results_pets.append(needed)
                    self.timings.append(timestamp)
            batch_frames.clear()
            batch_times.clear()
            if progress_cb:
                progress_cb(processed, total_samples)

        for frame_index, frame, fps, total_samples in self._iter_frames(
            video_path, interval
        ):
            batch_frames.append(frame)
            batch_times.append(round(frame_index / fps, 3) if fps else 0.0)
            if len(batch_frames) >= max(1, batch_size):
                flush()

        flush()
        if progress_cb:
            progress_cb(processed, max(processed, total_samples))
        return frames_pets, results_pets


class Recognizer_MP(Recognizer):
    """Тот же алгоритм, но кадры распознаются пакетами (быстрее на CPU/GPU)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = 0.6,
        batch_size: int = 8,
    ):
        super().__init__(model_path, confidence)
        self.batch_size = max(1, batch_size)

    def predict(
        self,
        video_path: str | Path,
        interval: float = 1,
        index: int = DOG_CLASS_ID,
        progress_cb: ProgressCallback | None = None,
        batch_size: int | None = None,
    ) -> tuple[list[np.ndarray], list[list]]:
        return super().predict(
            video_path,
            interval=interval,
            index=index,
            progress_cb=progress_cb,
            batch_size=batch_size or self.batch_size,
        )
