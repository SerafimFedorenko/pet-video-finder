"""Тесты CLIP-сравнения без загрузки настоящих моделей."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from comparison import ClipSimilarity, cosine_to_score

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class FakeEmbeddingModel:
    """Детерминированные векторы: красный/кот, зелёный/собака, остальное."""

    def __init__(self):
        self.calls: list[list] = []

    def encode(self, inputs, **kwargs):
        self.calls.append(list(inputs))
        return np.asarray([self._vector(item) for item in inputs], dtype=np.float32)

    @staticmethod
    def _vector(item) -> list[float]:
        if isinstance(item, str):
            text = item.lower()
            if "кот" in text:
                return [1.0, 0.0, 0.0]
            if "собак" in text:
                return [0.0, 1.0, 0.0]
            return [0.0, 0.0, 1.0]

        pixel = item.convert("RGB").getpixel((0, 0))
        if pixel == RED:
            return [1.0, 0.0, 0.0]
        if pixel == GREEN:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def solid(color) -> Image.Image:
    return Image.new("RGB", (8, 8), color)


class CosineToScoreTest(unittest.TestCase):
    def test_clamps_cosine_to_zero_and_one(self):
        self.assertEqual(cosine_to_score(-0.5), 0.0)
        self.assertEqual(cosine_to_score(0.5), 0.5)
        self.assertEqual(cosine_to_score(1.5), 1.0)


class ClipSimilarityTest(unittest.TestCase):
    def setUp(self):
        self.image_model = FakeEmbeddingModel()
        self.text_model = FakeEmbeddingModel()
        self.similarity = ClipSimilarity(
            image_model=self.image_model,
            text_model=self.text_model,
            batch_size=4,
        )

    def test_text_matches_expected_color(self):
        scores = self.similarity.predict_many_by_text(
            "рыжий кот", [solid(RED), solid(GREEN), solid(BLUE)]
        )

        self.assertEqual(scores, [1.0, 0.0, 0.0])

    def test_image_matches_expected_color(self):
        scores = self.similarity.predict_many_by_img(
            solid(RED), [solid(RED), solid(GREEN), solid(BLUE)]
        )

        self.assertEqual(scores, [1.0, 0.0, 0.0])

    def test_text_embedding_is_cached(self):
        self.similarity.predict_by_text("кот", solid(RED))
        self.similarity.predict_by_text("кот", solid(GREEN))

        self.assertEqual(len(self.text_model.calls), 1)
        self.assertEqual(len(self.image_model.calls), 2)

    def test_reference_embedding_is_cached_for_same_image(self):
        reference = solid(RED)
        self.similarity.predict_many_by_img(reference, [solid(RED), solid(GREEN)])
        self.similarity.predict_many_by_img(reference, [solid(BLUE)])

        # 1 вызов на эталон + 2 вызова на пачки сравниваемых изображений.
        self.assertEqual(len(self.image_model.calls), 3)

    def test_empty_batches_return_empty_lists(self):
        self.assertEqual(self.similarity.predict_many_by_text("кот", []), [])
        self.assertEqual(self.similarity.predict_many_by_img(solid(RED), []), [])
        self.assertEqual(self.image_model.calls, [])
        self.assertEqual(self.text_model.calls, [])


if __name__ == "__main__":
    unittest.main()
