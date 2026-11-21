import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from recognizer import DOG_CLASS_ID, Recognizer


def _result_with_boxes(boxes: np.ndarray):
    result = MagicMock()
    result.boxes.data.cpu.return_value.numpy.return_value = boxes
    return result


class TestRecognizer(unittest.TestCase):
    @patch("recognizer.YOLO")
    def test_initialization(self, MockYOLO):
        recognizer = Recognizer()
        self.assertIsNotNone(recognizer.model)
        called_path = Path(MockYOLO.call_args.args[0])
        self.assertEqual(called_path.name, "yolov8n.pt")

    @patch("recognizer.YOLO")
    def test_predict_img(self, MockYOLO):
        recognizer = Recognizer()
        boxes = np.array([[0, 0, 10, 10, 0.9, 16]])
        recognizer.model.return_value = [_result_with_boxes(boxes)]

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = recognizer.predict_img(image)

        self.assertEqual(results.shape, (1, 6))
        self.assertEqual(results[0][-1], 16)

    @patch("recognizer.YOLO")
    def test_predict_img_without_boxes(self, MockYOLO):
        recognizer = Recognizer()
        result = MagicMock()
        result.boxes = None
        recognizer.model.return_value = [result]

        self.assertEqual(recognizer.predict_img(np.zeros((10, 10, 3), np.uint8)).shape, (0, 6))

    def test_check_index(self):
        results = np.array([[0, 0, 10, 10, 0.9, 16], [10, 10, 20, 20, 0.8, 17]])
        filtered = Recognizer.check_index(results, DOG_CLASS_ID)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][-1], 16)

    @patch("recognizer.cv2.VideoCapture")
    @patch("recognizer.YOLO")
    def test_predict(self, MockYOLO, MockVideoCapture):
        capture = MockVideoCapture.return_value
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: (
            30.0 if prop == cv2.CAP_PROP_FPS else 1
        )
        capture.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))

        recognizer = Recognizer()
        recognizer.model.return_value = [
            _result_with_boxes(np.array([[0, 0, 10, 10, 0.9, 16]]))
        ]

        frames_pets, results_pets = recognizer.predict("dummy_path", interval=1)

        self.assertEqual(len(frames_pets), 1)
        self.assertEqual(len(results_pets), 1)
        self.assertEqual(results_pets[0][0][-1], 16)
        self.assertEqual(len(recognizer.timings), 1)
        capture.release.assert_called_once()

    @patch("recognizer.cv2.VideoCapture")
    @patch("recognizer.YOLO")
    def test_predict_raises_for_missing_video(self, MockYOLO, MockVideoCapture):
        MockVideoCapture.return_value.isOpened.return_value = False
        recognizer = Recognizer()
        with self.assertRaises(IOError):
            recognizer.predict("missing.mp4")


if __name__ == "__main__":
    unittest.main()
