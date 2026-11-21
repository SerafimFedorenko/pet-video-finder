"""Тесты операций над результатами: фильтр, сохранение, экспорт."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook
from PIL import Image

from controllers.analysis import FrameResult
from controllers.results import (
    export_results,
    filter_frames,
    format_timestamp,
    save_best_crops,
)


def make_result(score: float, best_crop_index: int = 0, colors=("red", "green")) -> FrameResult:
    crops = tuple(Image.new("RGB", (20, 20), color) for color in colors)
    return FrameResult(
        image=Image.new("RGB", (40, 40), "black"),
        crops=crops,
        best_crop_index=best_crop_index,
        score=score,
        timestamp=75.0,
    )


class FormatTimestampTest(unittest.TestCase):
    def test_formats_minutes_and_seconds(self):
        self.assertEqual(format_timestamp(75), "1 мин 15 c")
        self.assertEqual(format_timestamp(0), "0 мин 0 c")
        self.assertEqual(format_timestamp(-10), "0 мин 0 c")


class FilterFramesTest(unittest.TestCase):
    def test_keeps_frames_at_or_above_threshold(self):
        frames = [make_result(0.49), make_result(0.5), make_result(0.9)]

        selected = filter_frames(frames, 50)

        self.assertEqual([frame.score for frame in selected], [0.5, 0.9])

    def test_zero_threshold_keeps_everything(self):
        frames = [make_result(0.0), make_result(0.3)]

        self.assertEqual(filter_frames(frames, 0), frames)


class SaveBestCropsTest(unittest.TestCase):
    def test_saves_one_best_crop_per_frame(self):
        frames = [make_result(0.7, best_crop_index=1), make_result(0.8, best_crop_index=0)]

        with tempfile.TemporaryDirectory() as directory:
            saved = save_best_crops(frames, directory)
            files = sorted(Path(directory).glob("*.jpg"))

        self.assertEqual(saved, 2)
        self.assertEqual([file.name for file in files], ["image_1.jpg", "image_2.jpg"])

    def test_clamps_out_of_range_best_index(self):
        frames = [make_result(0.7, best_crop_index=99)]

        with tempfile.TemporaryDirectory() as directory:
            saved = save_best_crops(frames, directory)
            files = list(Path(directory).glob("*.jpg"))

        self.assertEqual(saved, 1)
        self.assertEqual(len(files), 1)

    def test_skips_frames_without_crops(self):
        frame = FrameResult(
            image=Image.new("RGB", (10, 10)),
            crops=(),
            best_crop_index=0,
            score=0.5,
            timestamp=0.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            saved = save_best_crops([frame], directory)

        self.assertEqual(saved, 0)


class ExportResultsTest(unittest.TestCase):
    def test_writes_timestamp_and_score(self):
        frames = [make_result(0.75), make_result(0.9)]

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "results.xlsx"
            export_results(frames, target)
            workbook = load_workbook(target)
            rows = list(workbook["Результаты"].values)

        self.assertEqual(rows[0], ("Время", "Сходство"))
        self.assertEqual(rows[1], ("1 мин 15 c", 75.0))
        self.assertEqual(rows[2], ("1 мин 15 c", 90.0))


if __name__ == "__main__":
    unittest.main()
