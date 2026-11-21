"""Сравнение изображений и текста через CLIP (sentence-transformers).

Для изображений используется CLIP ViT-B/32, для текста — мультиязычная
модель, которая отображает русские и английские описания в то же
пространство признаков CLIP. Итоговая оценка — косинусное сходство.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from threading import Lock

import numpy as np
from PIL import Image

# Transformers не должен подхватывать установленный TensorFlow с Keras 3:
# для CLIP достаточно PyTorch. Переменные нужно выставить до первого импорта
# sentence-transformers/transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from config import (  # noqa: E402  (после настройки переменных окружения)
    CAT_CLASS_ID,
    CLIP_BATCH_SIZE,
    CLIP_IMAGE_MODEL,
    CLIP_TEXT_MODEL,
    DOG_CLASS_ID,
)


def cosine_to_score(cosine: float) -> float:
    """Косинус -> оценка 0..1, которую показывает интерфейс.

    Отрицательные значения обрезаются: для пользователя «похожесть»
    не может быть отрицательной. Это не вероятность, а косинусное
    сходство, поэтому фильтр в процентах трактуется соответственно.
    """
    return float(np.clip(cosine, 0.0, 1.0))


class ClipSimilarity:
    """Считает косинусное сходство изображений и текста в пространстве CLIP.

    Модели загружаются лениво: для режима сравнения изображений нужна
    только CLIP-модель изображений, текстовый энкодер подключается при
    первом сравнении с описанием. Это заметно ускоряет запуск.
    """

    def __init__(
        self,
        image_model=None,
        text_model=None,
        image_model_name: str | None = None,
        text_model_name: str | None = None,
        device: str | None = None,
        batch_size: int = CLIP_BATCH_SIZE,
        cache_size: int = 128,
    ):
        self.image_model_name = image_model_name or CLIP_IMAGE_MODEL
        self.text_model_name = text_model_name or CLIP_TEXT_MODEL
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.cache_size = max(1, int(cache_size))

        self._image_model = image_model
        self._text_model = text_model
        self._model_lock = Lock()
        self._text_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._reference_image = None
        self._reference_embedding: np.ndarray | None = None

    # ---------- ленивая загрузка моделей ----------

    @property
    def image_model(self):
        if self._image_model is None:
            with self._model_lock:
                if self._image_model is None:
                    self._image_model = self._load_model(self.image_model_name)
        return self._image_model

    @property
    def text_model(self):
        if self._text_model is None:
            with self._model_lock:
                if self._text_model is None:
                    self._text_model = self._load_model(self.text_model_name)
        return self._text_model

    def _load_model(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name, device=self.device)

    # ---------- кодирование ----------

    def _encode(self, model, inputs: list) -> np.ndarray:
        embeddings = model.encode(
            inputs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array[np.newaxis, :]
        return array

    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        prepared = [image.convert("RGB") for image in images]
        return self._encode(self.image_model, prepared)

    def _text_embedding(self, text: str) -> np.ndarray:
        key = " ".join(str(text).split())
        cached = self._text_cache.get(key)
        if cached is not None:
            self._text_cache.move_to_end(key)
            return cached

        embedding = self._encode(self.text_model, [key])[0]
        self._text_cache[key] = embedding
        if len(self._text_cache) > self.cache_size:
            self._text_cache.popitem(last=False)
        return embedding

    def _reference_features(self, image: Image.Image) -> np.ndarray:
        if self._reference_image is not image:
            self._reference_image = image
            self._reference_embedding = self._encode_images([image])[0]
        return self._reference_embedding

    # ---------- публичный API ----------

    def predict_by_text(self, text: str, image: Image.Image) -> float:
        return self.predict_many_by_text(text, [image])[0]

    def predict_by_img(self, reference: Image.Image, image: Image.Image) -> float:
        return self.predict_many_by_img(reference, [image])[0]

    def predict_many_by_text(
        self, text: str, images: list[Image.Image]
    ) -> list[float]:
        """Сходство одного описания с несколькими изображениями."""
        if not images:
            return []

        text_embedding = self._text_embedding(text)
        image_embeddings = self._encode_images(images)
        cosines = image_embeddings @ text_embedding
        return [cosine_to_score(value) for value in cosines]

    def predict_many_by_img(
        self, reference: Image.Image, images: list[Image.Image]
    ) -> list[float]:
        """Сходство эталонного изображения с несколькими изображениями."""
        if not images:
            return []

        reference_embedding = self._reference_features(reference)
        image_embeddings = self._encode_images(images)
        cosines = image_embeddings @ reference_embedding
        return [cosine_to_score(value) for value in cosines]


def create_predictors() -> tuple[dict[int, ClipSimilarity], dict[int, ClipSimilarity]]:
    """Готовые предикторы для кошек и собак.

    CLIP не зависит от класса животного, поэтому одна модель используется
    сразу для обоих классов и для обоих режимов сравнения.
    """
    similarity = ClipSimilarity()
    text_predictors = {CAT_CLASS_ID: similarity, DOG_CLASS_ID: similarity}
    image_predictors = {CAT_CLASS_ID: similarity, DOG_CLASS_ID: similarity}
    return text_predictors, image_predictors
