"""Тесты бизнес-логики анализа видео (без загрузки YOLO и Keras)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from config import CAT_CLASS_ID, DOG_CLASS_ID, MODE_IMAGE, MODE_TEXT
from controllers.analysis import AnalysisRequest, VideoAnalysisController


class FakeYolo:
    """Подменяет Recognizer: возвращает заранее заданные кадры и боксы."""

    def __init__(self, frames=None, boxes=None, timings=None):
        self.frames = list(frames or [])
        self.boxes = list(boxes or [])
        self.timings = list(timings or [])
        self.calls: list[dict] = []
        self.predict_img_calls: list[np.ndarray] = []

    def predict(self, video_path, interval=1, index=DOG_CLASS_ID, progress_cb=None, **kwargs):
        self.calls.append(
            {
                "video_path": video_path,
                "interval": interval,
                "index": index,
            }
        )
        if progress_cb is not None:
            total = len(self.frames) or 1
            progress_cb(len(self.frames), total)
        return list(self.frames), list(self.boxes)

    def predict_img(self, img_cv2, confidence=None):
        self.predict_img_calls.append(img_cv2)
        return np.empty((0, 6))


class FakeTextPredictor:
    """Возвращает высокий балл только для кропа 60x60."""

    def __init__(self):
        self.calls: list[tuple[str, tuple[int, int]]] = []

    def predict_by_text(self, text: str, image: Image.Image) -> float:
        self.calls.append((text, image.size))
        return 0.9 if image.size == (60, 60) else 0.2


class FakeImagePredictor:
    def __init__(self, score: float = 0.7):
        self.score = score
        self.calls: list[tuple[Image.Image, Image.Image]] = []

    def predict_by_img(self, reference: Image.Image, image: Image.Image) -> float:
        self.calls.append((reference, image))
        return self.score


class BatchTextPredictor(FakeTextPredictor):
    """Проверяет, что контроллер использует пакетное сравнение."""

    def __init__(self):
        super().__init__()
        self.batch_calls: list[tuple[str, list[Image.Image]]] = []

    def predict_many_by_text(
        self, text: str, images: list[Image.Image]
    ) -> list[float]:
        self.batch_calls.append((text, list(images)))
        return [0.9 if image.size == (60, 60) else 0.2 for image in images]


def make_frame(height: int = 100, width: int = 100) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    return frame


def make_controller(yolo, text_predictor=None, image_predictor=None):
    return VideoAnalysisController(
        yolo=yolo,
        text_predictors={CAT_CLASS_ID: text_predictor or FakeTextPredictor()},
        image_predictors={CAT_CLASS_ID: image_predictor or FakeImagePredictor()},
    )


class AnalysisRequestTest(unittest.TestCase):
    def test_text_request_is_valid(self):
        request = AnalysisRequest(
            class_index=CAT_CLASS_ID, interval=1, mode=MODE_TEXT, description="кот"
        )
        request.validate()

    def test_unknown_mode_is_rejected(self):
        request = AnalysisRequest(
            class_index=CAT_CLASS_ID, interval=1, mode="video", description="кот"
        )
        with self.assertRaises(ValueError):
            request.validate()

    def test_non_positive_interval_is_rejected(self):
        request = AnalysisRequest(
            class_index=CAT_CLASS_ID, interval=0, mode=MODE_TEXT, description="кот"
        )
        with self.assertRaises(ValueError):
            request.validate()

    def test_empty_description_is_rejected(self):
        request = AnalysisRequest(
            class_index=CAT_CLASS_ID, interval=1, mode=MODE_TEXT, description="   "
        )
        with self.assertRaises(ValueError):
            request.validate()

    def test_image_mode_requires_reference(self):
        request = AnalysisRequest(
            class_index=CAT_CLASS_ID, interval=1, mode=MODE_IMAGE
        )
        with self.assertRaises(ValueError):
            request.validate()


class AnalyzeTest(unittest.TestCase):
    def test_text_mode_returns_best_crop_and_score(self):
        frame = make_frame()
        boxes = [[10, 10, 50, 50], [20, 20, 80, 80]]
        yolo = FakeYolo([frame], [boxes], timings=[1.5])
        predictor = FakeTextPredictor()
        controller = make_controller(yolo, text_predictor=predictor)
        progress: list[int] = []

        result = controller.analyze(
            "video.mp4",
            AnalysisRequest(
                class_index=CAT_CLASS_ID,
                interval=2,
                mode=MODE_TEXT,
                description="рыжий кот",
            ),
            progress_cb=progress.append,
        )

        self.assertEqual(len(result.frames), 1)
        found = result.frames[0]
        self.assertEqual([crop.size for crop in found.crops], [(40, 40), (60, 60)])
        self.assertEqual(found.best_crop_index, 1)
        self.assertAlmostEqual(found.score, 0.9)
        self.assertAlmostEqual(found.timestamp, 1.5)
        self.assertEqual(found.image.size, (100, 100))
        self.assertEqual(predictor.calls, [("рыжий кот", (40, 40)), ("рыжий кот", (60, 60))])
        self.assertEqual(progress[-1], 100)
        self.assertTrue(all(0 <= value <= 100 for value in progress))
        self.assertEqual(yolo.calls[0]["interval"], 2)
        self.assertEqual(yolo.calls[0]["index"], CAT_CLASS_ID)

    def test_uses_batch_scoring_when_available(self):
        frame = make_frame()
        yolo = FakeYolo([frame], [[[0, 0, 40, 40], [20, 20, 80, 80]]])
        predictor = BatchTextPredictor()
        controller = make_controller(yolo, text_predictor=predictor)

        result = controller.analyze(
            "video.mp4",
            AnalysisRequest(
                class_index=CAT_CLASS_ID,
                interval=1,
                mode=MODE_TEXT,
                description="кот",
            ),
        )

        self.assertEqual(len(predictor.batch_calls), 1)
        self.assertEqual(predictor.calls, [])
        self.assertEqual(result.frames[0].best_crop_index, 1)
        self.assertAlmostEqual(result.frames[0].score, 0.9)

    def test_image_mode_uses_reference_image(self):
        frame = make_frame()
        reference = Image.new("RGB", (50, 50), "red")
        yolo = FakeYolo([frame], [[[0, 0, 50, 50]]], timings=[3.0])
        predictor = FakeImagePredictor(score=0.8)
        controller = make_controller(yolo, image_predictor=predictor)
        progress: list[int] = []

        result = controller.analyze(
            "video.mp4",
            AnalysisRequest(
                class_index=CAT_CLASS_ID,
                interval=1,
                mode=MODE_IMAGE,
                reference_image=reference,
            ),
            progress_cb=progress.append,
        )

        self.assertEqual(len(result.frames), 1)
        self.assertAlmostEqual(result.frames[0].score, 0.8)
        self.assertEqual(result.frames[0].best_crop_index, 0)
        self.assertIs(predictor.calls[0][0], reference)
        self.assertEqual(progress[-1], 100)

    def test_no_frames_returns_empty_result(self):
        controller = make_controller(FakeYolo())
        progress: list[int] = []

        result = controller.analyze(
            "video.mp4",
            AnalysisRequest(
                class_index=CAT_CLASS_ID, interval=1, mode=MODE_TEXT, description="кот"
            ),
            progress_cb=progress.append,
        )

        self.assertEqual(result.frames, ())
        self.assertEqual(progress[-1], 100)
        self.assertTrue(all(0 <= value <= 100 for value in progress))

    def test_boxes_outside_frame_are_clamped(self):
        frame = make_frame()
        yolo = FakeYolo([frame], [[[-5, -5, 200, 200]]], timings=[0.0])
        controller = make_controller(yolo)

        result = controller.analyze(
            "video.mp4",
            AnalysisRequest(
                class_index=CAT_CLASS_ID, interval=1, mode=MODE_TEXT, description="кот"
            ),
        )

        self.assertEqual(result.frames[0].crops[0].size, (100, 100))

    def test_empty_boxes_produce_placeholder_crop(self):
        frame = make_frame()
        yolo = FakeYolo([frame], [[]], timings=[0.0])
        controller = make_controller(yolo)

        result = controller.analyze(
            "video.mp4",
            AnalysisRequest(
                class_index=CAT_CLASS_ID, interval=1, mode=MODE_TEXT, description="кот"
            ),
        )

        self.assertEqual(len(result.frames[0].crops), 1)
        self.assertEqual(result.frames[0].crops[0].size, (224, 224))

    def test_missing_predictor_raises_value_error(self):
        frame = make_frame()
        yolo = FakeYolo([frame], [[[0, 0, 50, 50]]], timings=[0.0])
        controller = make_controller(yolo)

        with self.assertRaises(ValueError):
            controller.analyze(
                "video.mp4",
                AnalysisRequest(
                    class_index=DOG_CLASS_ID,
                    interval=1,
                    mode=MODE_TEXT,
                    description="собака",
                ),
            )


class FakeCapture:
    def __init__(self, fps: float = 10.0, frame_count: int = 100):
        self.fps = fps
        self.frame_count = frame_count
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        return 0

    def read(self):
        return True, make_frame(20, 20)

    def release(self):
        self.released = True


class EstimateTimeTest(unittest.TestCase):
    def test_estimate_uses_fps_and_interval(self):
        capture = FakeCapture(fps=10.0, frame_count=100)
        yolo = FakeYolo()
        controller = make_controller(yolo)
        request = AnalysisRequest(
            class_index=CAT_CLASS_ID, interval=2, mode=MODE_TEXT, description="кот"
        )

        with patch("controllers.analysis.cv2.VideoCapture", return_value=capture):
            estimated = controller.estimate_time("video.mp4", request)

        self.assertGreater(estimated, 0)
        self.assertTrue(capture.released)
        self.assertEqual(len(yolo.predict_img_calls), 2)

    def test_unreadable_video_raises(self):
        class BrokenCapture(FakeCapture):
            def read(self):
                return False, None

        controller = make_controller(FakeYolo())
        request = AnalysisRequest(
            class_index=CAT_CLASS_ID, interval=1, mode=MODE_TEXT, description="кот"
        )

        with patch(
            "controllers.analysis.cv2.VideoCapture", return_value=BrokenCapture()
        ):
            with self.assertRaises(IOError):
                controller.estimate_time("video.mp4", request)


if __name__ == "__main__":
    unittest.main()
