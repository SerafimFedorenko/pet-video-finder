"""Экспериментальное обучение модели сравнения пары изображений.

ВНИМАНИЕ: приложение использует другие веса — `dogs_img_model.keras` и
`cats_img_model.keras` являются VGG16-экстракторами признаков (см.
test_network.py), а не этой парной моделью. Скрипт оставлен для экспериментов.

Запуск:
    python network_train_img.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from keras import Input, Model
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import Concatenate, Dense, Dropout, Flatten, BatchNormalization
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from keras.optimizers import Adam
from keras.utils import img_to_array, load_img
import pandas as pd
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dogs data files"

INPUT_SHAPE = (224, 224, 3)


def load_and_preprocess_image(image_path, target_size=INPUT_SHAPE):
    image = load_img(image_path, target_size=target_size)
    return preprocess_input(img_to_array(image, dtype="float32"))


def create_model() -> Model:
    base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    base_model.trainable = False

    inputs = Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=False)
    outputs = Flatten()(x)
    feature_extractor = Model(inputs, outputs)

    input_image1 = Input(shape=INPUT_SHAPE)
    input_image2 = Input(shape=INPUT_SHAPE)

    features1 = feature_extractor(input_image1)
    features2 = feature_extractor(input_image2)

    combined = Concatenate()([features1, features2])
    x = BatchNormalization()(combined)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[input_image1, input_image2], outputs=output)
    model.compile(
        optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()]
    )
    return model


def _load_pairs(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = data.columns
    x1, x2, y = [], [], []
    for i in range(len(data[columns[0]])):
        x1.append(load_and_preprocess_image(data[columns[0]][i]))
        x2.append(load_and_preprocess_image(data[columns[1]][i]))
        y.append(int(data[columns[2]][i]))
    return np.array(x1), np.array(x2), np.array(y)


def main() -> None:
    train_data = pd.read_excel(DATA_DIR / "train.xlsx")
    columns = train_data.columns
    indices = list(range(len(train_data[columns[0]])))

    model = create_model()
    model.summary()

    for iteration in range(10):
        print(f"iteration {iteration}")
        sample = random.sample(indices, max(1, len(indices) // 6))
        subset = train_data.iloc[sample]
        x1, x2, y = _load_pairs(subset)
        model.fit([x1, x2], y, validation_split=0.2, batch_size=16, epochs=5)

        model.save(str(PROJECT_ROOT / "dogs_pair_model.keras"))

    test_data = pd.read_excel(DATA_DIR / "test.xlsx")
    x1_test, x2_test, y_test = _load_pairs(test_data)
    predictions = model.predict([x1_test, x2_test], verbose=0)
    y_pred = np.rint(predictions).astype(int).ravel()
    print(f"accuracy: {accuracy_score(y_test, y_pred)}")


if __name__ == "__main__":
    main()
