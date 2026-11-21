"""Проверка моделей сравнения изображений на отложенной выборке.

Запуск: python network_test_img.py [dogs|cats]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score

import networks

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pet", choices=["dogs", "cats"], nargs="?", default="dogs")
    args = parser.parse_args()

    folder = "dogs data files" if args.pet == "dogs" else "cats data files"
    predictor = (
        networks.DogsImgPredictor() if args.pet == "dogs" else networks.CatsImgPredictor()
    )

    test_data = pd.read_excel(PROJECT_ROOT / folder / "test.xlsx")
    columns = test_data.columns

    labels: list[int] = []
    predictions: list[float] = []
    for i in range(len(test_data[columns[0]])):
        first = Image.open(test_data[columns[0]][i])
        second = Image.open(test_data[columns[1]][i])
        predictions.append(predictor.predict_by_img(first, second))
        labels.append(int(test_data[columns[2]][i]))
        print(
            f"{test_data[columns[0]][i]} | {test_data[columns[1]][i]} | "
            f"{predictions[-1]:.4f}"
        )

    threshold_predictions = [int(round(value)) for value in predictions]
    print(f"accuracy: {accuracy_score(labels, threshold_predictions)}")


if __name__ == "__main__":
    main()
