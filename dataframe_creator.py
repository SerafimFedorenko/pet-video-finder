"""Построение таблиц с парами изображений для обучения моделей сравнения."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent


def remove_permutation_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет строки-перестановки (A,B) и (B,A)."""
    df_copy = df.copy()
    df_copy["sorted_values"] = df_copy.apply(lambda row: tuple(sorted(row)), axis=1)
    df_copy.drop_duplicates(subset=["sorted_values"], inplace=True)
    df_copy.drop(columns=["sorted_values"], inplace=True)
    return df_copy


class DataFrameCreator:
    def create_dataframe(self, dataframe_name: str, folder_name: str) -> Path:
        """Создаёт <folder_name>/<dataframe_name>.xlsx с парами изображений.

        Ожидаемая структура: <folder_name>/frames/<объект>/<dataframe_name>/*.jpg
        """
        path_frames = PROJECT_ROOT / folder_name / "frames"
        if not path_frames.is_dir():
            # Сохраняем прежнее поведение: os.listdir падал с FileNotFoundError.
            raise FileNotFoundError(f"Не найдена папка с кадрами: {path_frames}")

        dirs = [path_frames / name for name in os.listdir(path_frames)]

        train_data_1: list[str] = []
        train_data_2: list[str] = []
        train_data_3: list[str] = []

        for dir_to_1 in dirs:
            for dir_to_2 in dirs:
                dir_train_1 = dir_to_1 / dataframe_name
                dir_train_2 = dir_to_2 / dataframe_name
                if not dir_train_1.is_dir() or not dir_train_2.is_dir():
                    continue
                train_files_1 = os.listdir(dir_train_1)
                train_files_2 = os.listdir(dir_train_2)
                for file_1 in train_files_1:
                    for file_2 in train_files_2:
                        if file_1 == file_2:
                            continue
                        train_data_1.append(str(dir_train_1 / file_1))
                        train_data_2.append(str(dir_train_2 / file_2))
                        train_data_3.append("1" if dir_to_1 == dir_to_2 else "0")

        train_data = pd.DataFrame(
            {"First": train_data_1, "Second": train_data_2, "Res": train_data_3}
        )
        data_not_duplicated = remove_permutation_duplicates(train_data)

        positive_examples = data_not_duplicated[data_not_duplicated["Res"] == "1"]
        negative_examples = data_not_duplicated[data_not_duplicated["Res"] == "0"]

        min_count = min(len(positive_examples), len(negative_examples))
        positive_examples_sampled = positive_examples.sample(
            min_count, random_state=42
        )
        negative_examples_sampled = negative_examples.sample(
            min_count, random_state=42
        )

        balanced_data = pd.concat(
            [positive_examples_sampled, negative_examples_sampled]
        ).reset_index(drop=True)

        output_path = PROJECT_ROOT / folder_name / f"{dataframe_name}.xlsx"
        balanced_data.to_excel(output_path, index=False)
        return output_path


def main() -> None:
    creator = DataFrameCreator()
    for folder in ("cats data files", "dogs data files"):
        for dataframe_name in ("train", "test"):
            try:
                path = creator.create_dataframe(dataframe_name, folder)
                print(f"Создан файл: {path}")
            except FileNotFoundError as exc:
                print(f"Пропущено: {exc}")


if __name__ == "__main__":
    main()
