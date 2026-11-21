"""Создание xlsx-таблицы «описание -> изображение» из текстового файла."""

from __future__ import annotations

import argparse
from pathlib import Path

import openpyxl

PROJECT_ROOT = Path(__file__).resolve().parent


def build_excel(source: Path, target: Path) -> None:
    with source.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.cell(row=1, column=1, value="описание")
    sheet.cell(row=1, column=2, value="изображение")

    for i, line in enumerate(lines, start=2):
        sheet.cell(row=i, column=1, value=line.strip())
        sheet.cell(row=i, column=2, value=f"{i - 1}.jpg")

    workbook.save(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=str(PROJECT_ROOT / "Descriptions_cats2.txt"))
    parser.add_argument("target", nargs="?", default=str(PROJECT_ROOT / "descriptions_cats.xlsx"))
    args = parser.parse_args()
    build_excel(Path(args.source), Path(args.target))
    print(f"Создан файл: {args.target}")


if __name__ == "__main__":
    main()
