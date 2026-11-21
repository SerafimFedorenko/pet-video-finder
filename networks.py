"""Обученные модели: сопоставление кадра с описанием и с эталонным изображением."""

from __future__ import annotations

import pickle
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras.applications import vgg16
from keras.utils import img_to_array

from config import IMAGE_SIZE, PROJECT_ROOT

# Все пути проекта считаются от каталога config.py,
# поэтому приложение можно запускать из любого рабочего каталога.


def _model_path(file_name: str) -> Path:
    path = PROJECT_ROOT / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл модели: {path}. "
            "Положите обученные модели в каталог проекта."
        )
    return path


def _load_text_vectorizer(
    file_name: str, descriptions_file: Path
) -> keras.layers.TextVectorization:
    """Восстанавливает слой TextVectorization, сохранённый рядом с моделью."""
    with open(_model_path(file_name), "rb") as f:
        from_disk = pickle.load(f)

    config = dict(from_disk.get("config") or {})
    vocabulary = from_disk.get("vocabulary") or config.get("vocabulary")

    if vocabulary:
        # Словарь нужно передать прямо в конструктор: одного from_config
        # недостаточно, иначе lookup-таблица остаётся неинициализированной.
        config["vocabulary"] = list(vocabulary)
        vectorizer = keras.layers.TextVectorization.from_config(config)
        if from_disk.get("weights"):
            vectorizer.set_weights(from_disk["weights"])
        return vectorizer

    # Старый формат pkl: словарь не сохранили. Восстанавливаем его по тем же
    # описаниям, на которых обучалась модель (порядок детерминирован).
    if not descriptions_file.exists():
        raise ValueError(
            f"В {file_name} нет словаря, а файл {descriptions_file} для его "
            "восстановления не найден. Переобучите модель network_train_text.py."
        )
    data = pd.read_excel(descriptions_file)
    descriptions = data["описание"].astype(str).tolist()

    vectorizer = keras.layers.TextVectorization(
        max_tokens=config.get("max_tokens", 1000),
        standardize=config.get("standardize", "lower_and_strip_punctuation"),
        split=config.get("split", "whitespace"),
        ngrams=config.get("ngrams"),
        output_mode=config.get("output_mode", "int"),
        output_sequence_length=config.get("output_sequence_length", 100),
    )
    # В network_train_text.py к описаниям добавлялись негативные примеры,
    # то есть фактически каждый текст встречался дважды.
    vectorizer.adapt(descriptions + descriptions)

    expected_size = config.get("vocabulary_size")
    actual_size = len(vectorizer.get_vocabulary())
    if expected_size and expected_size != actual_size:
        raise ValueError(
            f"Не удалось восстановить словарь {file_name}: ожидалось "
            f"{expected_size} слов, получено {actual_size}."
        )
    return vectorizer


def _prepare_image(image) -> np.ndarray:
    """PIL.Image/np.ndarray -> float32 массив (224, 224, 3) в диапазоне 0..255."""
    if hasattr(image, "convert"):  # PIL.Image
        image = image.convert("RGB")
        if image.size != IMAGE_SIZE:
            image = image.resize(IMAGE_SIZE)
        return img_to_array(image, dtype="float32")

    array = np.asarray(image, dtype="float32")
    if array.ndim == 4:  # на всякий случай: (1, H, W, C)
        array = array[0]
    if array.shape[:2] != IMAGE_SIZE:
        array = np.asarray(
            keras.utils.array_to_img(array).resize(IMAGE_SIZE), dtype="float32"
        )
    return array


class _TextPredictor:
    """Общая реализация для текстовых моделей кошек и собак."""

    model_file = ""
    vectorizer_file = ""
    descriptions_file = ""

    def __init__(self):
        self.model = keras.models.load_model(
            _model_path(self.model_file), compile=False
        )
        self.vectorizer = _load_text_vectorizer(
            self.vectorizer_file, PROJECT_ROOT / self.descriptions_file
        )

    def predict_by_text(self, text: str, image) -> float:
        # Модель обучалась на изображениях, делённых на 255
        # (см. network_train_text.py), поэтому здесь та же предобработка.
        image_array = _prepare_image(image) / 255.0
        text_sequence = np.asarray(self.vectorizer([str(text)]))
        res = self.model.predict(
            [np.expand_dims(image_array, axis=0), text_sequence],
            verbose=0,
        )
        return float(res[0][0])


class _ImagePredictor:
    """Общая реализация для моделей сравнения изображений."""

    model_file = ""

    def __init__(self):
        self.model = keras.models.load_model(
            _model_path(self.model_file), compile=False
        )

    @staticmethod
    def _features(model, image) -> np.ndarray:
        # Модели — это VGG16 без полносвязной части (см. test_network.py),
        # поэтому используется стандартная предобработка VGG16 (caffe mode).
        image_array = vgg16.preprocess_input(_prepare_image(image))
        return model.predict(
            np.expand_dims(image_array, axis=0), verbose=0
        ).flatten()

    def predict_by_img(self, image_true, image_compare) -> float:
        features1 = self._features(self.model, image_true)
        features2 = self._features(self.model, image_compare)
        # Косинусное сходство: 1 - косинусное расстояние.
        norm = np.linalg.norm(features1) * np.linalg.norm(features2)
        if norm == 0:
            return 0.0
        return float(np.dot(features1, features2) / norm)


class DogsTextPredictor(_TextPredictor):
    model_file = "dogs_text_model.keras"
    vectorizer_file = "tv_layer_doges.pkl"
    descriptions_file = "descriptions_dogs.xlsx"


class CatsTextPredictor(_TextPredictor):
    model_file = "cats_text_model.keras"
    vectorizer_file = "tv_layer_cats.pkl"
    descriptions_file = "descriptions_cats.xlsx"


class DogsImgPredictor(_ImagePredictor):
    model_file = "dogs_img_model.keras"


class CatsImgPredictor(_ImagePredictor):
    model_file = "cats_img_model.keras"
