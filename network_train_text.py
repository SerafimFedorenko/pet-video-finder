"""Обучение модели «описание <-> кадр» для кошек или собак.

Запуск:
    python network_train_text.py cats
    python network_train_text.py dogs

Результат: <pet>_text_model.keras и tv_layer_<pet>.pkl в каталоге проекта.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import VGG16
from keras.layers import (
    LSTM,
    BatchNormalization,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    concatenate,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_SIZE = (224, 224)
MAX_TOKENS = 1000
SEQUENCE_LENGTH = 100


def load_images(paths: list[Path]) -> np.ndarray:
    images = []
    for path in paths:
        try:
            image = keras.utils.load_img(path, target_size=IMAGE_SIZE)
            images.append(keras.utils.img_to_array(image))
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка загрузки изображения {path}: {exc}")
    return np.array(images, dtype="float32") / 255.0


def build_dataset(pet: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_excel(PROJECT_ROOT / f"descriptions_{pet}.xlsx")
    descriptions = data["описание"].astype(str).tolist()

    image_dir = PROJECT_ROOT / pet
    file_paths = [image_dir / str(name) for name in data["изображение"]]

    labels = [1] * len(descriptions)
    # Негативные примеры: то же описание, но случайное чужое изображение.
    for i in range(len(descriptions)):
        candidates = [p for j, p in enumerate(file_paths) if j != i]
        if candidates:
            descriptions.append(descriptions[i])
            file_paths.append(random.choice(candidates))
            labels.append(0)

    vectorizer = keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
    )
    vectorizer.adapt(descriptions)
    text_data = np.asarray(vectorizer(descriptions))
    image_data = load_images(file_paths)

    pickle.dump(
        {
            "config": vectorizer.get_config(),
            # get_config() не содержит восстановленного словаря, сохраняем явно.
            "vocabulary": vectorizer.get_vocabulary(),
            "weights": vectorizer.get_weights(),
        },
        open(PROJECT_ROOT / f"tv_layer_{pet}.pkl", "wb"),
    )
    return image_data, text_data, np.array(labels)


def build_model() -> keras.Model:
    image_input = Input(shape=IMAGE_SIZE + (3,))
    text_input = Input(shape=(SEQUENCE_LENGTH,))

    vgg = VGG16(weights="imagenet", include_top=False)
    vgg.trainable = False
    image_features = Flatten()(vgg(image_input))
    image_features = BatchNormalization()(image_features)
    image_features = Dropout(0.5)(image_features)

    text_embedding = Embedding(input_dim=MAX_TOKENS, output_dim=100)(text_input)
    text_features = LSTM(256, return_sequences=True)(text_embedding)
    text_features = LSTM(256)(text_features)

    combined = concatenate([image_features, text_features])
    combined = Dense(256, activation="relu")(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.5)(combined)
    outputs = Dense(1, activation="sigmoid")(combined)

    model = keras.Model(inputs=[image_input, text_input], outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pet", choices=["cats", "dogs"], nargs="?", default="cats")
    args = parser.parse_args()

    image_data, text_data, labels = build_dataset(args.pet)
    print(f"Примеров: {len(labels)} (изображения: {image_data.shape})")

    (
        image_train,
        image_val,
        text_train,
        text_val,
        y_train,
        y_val,
    ) = train_test_split(
        image_data, text_data, labels, test_size=0.1, random_state=42
    )

    model = build_model()
    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint(
        str(PROJECT_ROOT / f"{args.pet}_best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    model.fit(
        [image_train, text_train],
        y_train,
        validation_data=([image_val, text_val], y_val),
        epochs=20,
        batch_size=8,
        callbacks=[checkpoint],
    )

    model.save(str(PROJECT_ROOT / f"{args.pet}_text_model.keras"))


if __name__ == "__main__":
    main()
