"""Сравнение изображений по признакам VGG16 (косинусное расстояние).

Запуск: python test_network.py [dogs|cats]
Скрипт также сохраняет экстрактор признаков: <pet>_img_model.keras.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

PROJECT_ROOT = Path(__file__).resolve().parent


def load_image(path) -> np.ndarray:
    image = keras.utils.load_img(path, target_size=(224, 224))
    array = keras.utils.img_to_array(image, dtype="float32")
    return preprocess_input(np.expand_dims(array, axis=0))


def extract_features(model, path) -> np.ndarray:
    return model.predict(load_image(path), verbose=0)[0].flatten()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pet", choices=["dogs", "cats"], nargs="?", default="dogs")
    args = parser.parse_args()

    folder = "dogs data files" if args.pet == "dogs" else "cats data files"
    model = VGG16(weights="imagenet", include_top=False)
    model.save(str(PROJECT_ROOT / f"{args.pet}_img_model.keras"))

    test_data = pd.read_excel(PROJECT_ROOT / folder / "test.xlsx")
    columns = test_data.columns

    y_pred = []
    for i in range(len(test_data[columns[0]])):
        features1 = extract_features(model, test_data[columns[0]][i])
        features2 = extract_features(model, test_data[columns[1]][i])
        # 1 - косинусное расстояние = косинусное сходство.
        denominator = np.linalg.norm(features1) * np.linalg.norm(features2)
        similarity = (
            float(np.dot(features1, features2) / denominator) if denominator else 0.0
        )
        y_pred.append(similarity)
        print(
            f"{test_data[columns[0]][i]} | {test_data[columns[1]][i]} | {similarity:.4f}"
        )


if __name__ == "__main__":
    main()
