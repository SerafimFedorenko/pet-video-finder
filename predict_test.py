"""Сравнение скорости последовательной и пакетной обработки видео.

Запуск:
    python predict_test.py [путь_к_видео] [dogs|cats]

По умолчанию берётся первое видео из test_data/test_videos.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import recognizer

PROJECT_ROOT = Path(__file__).resolve().parent


def _default_video() -> Path | None:
    videos = sorted((PROJECT_ROOT / "test_data" / "test_videos").glob("*.mp4"))
    return videos[0] if videos else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?")
    parser.add_argument("pet", choices=["dogs", "cats"], nargs="?", default="dogs")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()

    video_path = Path(args.video) if args.video else _default_video()
    if video_path is None or not video_path.exists():
        raise SystemExit("Не найден видеофайл. Укажите путь аргументом.")
    index = recognizer.DOG_CLASS_ID if args.pet == "dogs" else recognizer.CAT_CLASS_ID

    yolo_linear = recognizer.Recognizer()
    start = time.time()
    _, results_linear = yolo_linear.predict(
        video_path, index=index, interval=args.interval
    )
    linear_time = time.time() - start
    print(f"Линейная обработка: {linear_time:.2f} с, кадров: {len(results_linear)}")
    print(f"Тайминги: {yolo_linear.timings}")

    yolo_mp = recognizer.Recognizer_MP()
    start = time.time()
    _, results_mp = yolo_mp.predict(video_path, index=index, interval=args.interval)
    mp_time = time.time() - start
    print(f"Пакетная обработка: {mp_time:.2f} с, кадров: {len(results_mp)}")
    print(f"Тайминги: {yolo_mp.timings}")


if __name__ == "__main__":
    main()
