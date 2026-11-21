"""Общие константы и пути проекта."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Индексы классов COCO, с которыми работает приложение.
CAT_CLASS_ID = 15
DOG_CLASS_ID = 16

# Размер изображения, который ожидают Keras-модели.
IMAGE_SIZE = (224, 224)

MODE_TEXT = "text"
MODE_IMAGE = "image"

# Сравнение кадров: "clip" — CLIP через sentence-transformers (по умолчанию),
# "legacy" — старые обученные Keras-модели на VGG16.
COMPARISON_BACKEND = os.environ.get("PET_FINDER_BACKEND", "clip").strip().lower()

# Раздельные модели: CLIP для изображений и мультиязычный текстовый энкодер,
# отображённый в то же пространство признаков (понимает русские описания).
CLIP_IMAGE_MODEL = os.environ.get(
    "PET_FINDER_CLIP_IMAGE_MODEL", "clip-ViT-B-32"
)
CLIP_TEXT_MODEL = os.environ.get(
    "PET_FINDER_CLIP_TEXT_MODEL",
    "sentence-transformers/clip-ViT-B-32-multilingual-v1",
)
CLIP_BATCH_SIZE = int(os.environ.get("PET_FINDER_CLIP_BATCH_SIZE", "32"))
