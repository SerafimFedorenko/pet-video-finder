"""Проверка окружения: импорты, модели и один кадр через YOLO.

Запуск:
    .venv\\Scripts\\python.exe check_install.py
    .venv\\Scripts\\python.exe check_install.py --legacy   # старые Keras-модели

По умолчанию проверяется основной backend сравнения — CLIP.
"""

from __future__ import annotations

import os
import sys

# Меньше служебных сообщений TensorFlow в консоли.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Иначе Transformers попытается импортировать TensorFlow и упадёт на Keras 3.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    print("Проверка импортов...")
    import cv2
    import keras
    import sentence_transformers
    import tensorflow as tf
    import torch
    import transformers
    import ultralytics

    print(f"  tensorflow {tf.__version__}")
    print(f"  keras      {keras.__version__}")
    print(f"  torch      {torch.__version__}")
    print(f"  ultralytics{ultralytics.__version__}")
    print(f"  opencv     {cv2.__version__}")
    print(f"  sentence-transformers {sentence_transformers.__version__}")
    print(f"  transformers {transformers.__version__}")

    test_image_path = PROJECT_ROOT / "doge.jpg"
    if test_image_path.exists():
        image = Image.open(test_image_path).convert("RGB")
    else:
        image = Image.new("RGB", (224, 224), color=(120, 120, 120))
    print(f"Тестовое изображение: {test_image_path if test_image_path.exists() else 'заглушка'}")

    print("Загрузка CLIP...")
    from comparison import ClipSimilarity

    clip = ClipSimilarity()
    text_score = clip.predict_by_text("собака на прогулке", image)
    image_score = clip.predict_by_img(image, image)
    print(f"  текст:        {text_score:.4f}")
    print(f"  изображение:  {image_score:.4f}")
    if image_score <= 0:
        raise RuntimeError("CLIP вернул нулевое сходство изображения с самим собой.")

    if "--legacy" in sys.argv:
        print("Загрузка старых Keras-моделей...")
        import networks

        print(f"  dogs_text: {networks.DogsTextPredictor().predict_by_text('собака на прогулке', image):.4f}")
        print(f"  cats_text: {networks.CatsTextPredictor().predict_by_text('кошка на прогулке', image):.4f}")
        print(f"  dogs_img:  {networks.DogsImgPredictor().predict_by_img(image, image):.4f}")
        print(f"  cats_img:  {networks.CatsImgPredictor().predict_by_img(image, image):.4f}")

    print("Проверка YOLO на изображении...")
    import recognizer

    yolo = recognizer.Recognizer()
    boxes = yolo.predict_img(np.asarray(image)[:, :, ::-1])
    print(f"  найдено объектов: {len(boxes)}")

    print("Готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
