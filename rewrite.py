"""Удаляет первое слово в каждой строке текстового файла.

ПРИМЕЧАНИЕ: в исходной версии функция называется remove_second_word,
но удаляет именно первое слово (del words[0]). Имя сохранено для совместимости.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def remove_second_word(filename1: str | Path, filename2: str | Path) -> None:
    with open(filename1, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open(filename2, "w", encoding="utf-8") as file:
        for line in lines:
            words = line.split()
            if words:
                del words[0]
            file.write(" ".join(words) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", default=str(PROJECT_ROOT / "Descriptions_cats.txt")
    )
    parser.add_argument(
        "target", nargs="?", default=str(PROJECT_ROOT / "Descriptions_cats2.txt")
    )
    args = parser.parse_args()
    remove_second_word(args.source, args.target)
    print(f"Записан файл: {args.target}")


if __name__ == "__main__":
    main()
