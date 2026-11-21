"""Нарезка обучающих кадров из видео с помощью YOLO.

Запускается вручную:
    python frame_cutter.py
Результат складывается в <папка>\\frames\\*_framesN.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import CAT_CLASS_ID, DOG_CLASS_ID, PROJECT_ROOT


class FrameCutter:
    def __init__(self, model_path: str | Path | None = None):
        model_file = Path(model_path) if model_path else PROJECT_ROOT / "yolov8n.pt"
        self.model = YOLO(str(model_file))

    def predict_img(self, img_cv2, confidence: float = 0.8):
        results = self.model(img_cv2, save=False, conf=confidence, verbose=False)
        return results[0].boxes.data

    @staticmethod
    def check_index(results, index: int):
        return [result for result in results if int(result[5]) == index]

    def cut_video(
        self,
        video_path: str | Path,
        interval: float = 1,
        index: int = DOG_CLASS_ID,
        max_frames: int = 15,
    ) -> list:
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            print(f"Не удалось открыть видео: {video_path}")
            return []

        fps = video.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"fps: {fps}")
        frames_pets = []
        frame_index = 0
        step = max(1, int(round(fps * max(interval, 0.001))))
        try:
            while True:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = video.read()
                if not ret:
                    break
                result_needed = self.check_index(self.predict_img(frame), index)
                if result_needed:
                    res = result_needed[0]
                    cropped = frame[int(res[1]): int(res[3]), int(res[0]): int(res[2])]
                    if cropped.size:
                        frames_pets.append(cropped)
                if len(frames_pets) >= max_frames:
                    break
                frame_index += step
        finally:
            video.release()
        return frames_pets

    @staticmethod
    def split_and_save_images(images: list, save_dir: str | Path) -> None:
        num_images = len(images)
        num_test = int(num_images * 0.2)
        num_train = num_images - num_test

        random.shuffle(images)

        save_dir_test = Path(save_dir) / "test"
        save_dir_train = Path(save_dir) / "train"
        save_dir_test.mkdir(parents=True, exist_ok=True)
        save_dir_train.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(images):
            directory = save_dir_test if i < num_test else save_dir_train
            cv2.imwrite(str(directory / f"image_{i}.jpg"), image)


def _collect_videos(directory: Path) -> list[Path]:
    return [
        Path(root) / name
        for root, _dirs, files in os.walk(directory)
        for name in files
        if (Path(root) / name).suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}
    ]


def main() -> None:
    definitions = (
        ("dogs data files", DOG_CLASS_ID, "dog_frames"),
        ("cats data files", CAT_CLASS_ID, "cat_frames"),
    )
    cutter = FrameCutter()
    for folder, class_index, prefix in definitions:
        videos = _collect_videos(PROJECT_ROOT / folder / "videos")
        if not videos:
            print(f"Нет видео в {PROJECT_ROOT / folder / 'videos'}")
            continue
        for counter, video_path in enumerate(videos):
            frames = cutter.cut_video(video_path, interval=0.3, index=class_index)
            save_dir = PROJECT_ROOT / folder / "frames" / f"{prefix}{counter}"
            cutter.split_and_save_images(frames, save_dir)
            print(f"{video_path.name}: сохранено кадров {len(frames)} -> {save_dir}")


if __name__ == "__main__":
    main()
