"""Операции над готовыми результатами: фильтр, сохранение, экспорт."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from openpyxl import Workbook

from .analysis import FrameResult


def format_timestamp(seconds: float) -> str:
    minutes, remainder = divmod(max(0.0, float(seconds)), 60)
    return f"{int(minutes)} мин {round(remainder)} c"


def filter_frames(
    frames: Sequence[FrameResult], threshold_percent: float
) -> list[FrameResult]:
    """Оставляет кадры, у которых сходство не ниже порога (в процентах)."""
    return [frame for frame in frames if frame.score * 100 >= threshold_percent]


def save_best_crops(frames: Sequence[FrameResult], directory: str | Path) -> int:
    """Сохраняет лучший кроп каждого кадра. Возвращает число файлов."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    saved = 0
    for index, frame in enumerate(frames, start=1):
        if not frame.crops:
            continue
        best = min(max(frame.best_crop_index, 0), len(frame.crops) - 1)
        frame.crops[best].save(target / f"image_{index}.jpg")
        saved += 1
    return saved


def export_results(frames: Sequence[FrameResult], file_path: str | Path) -> Path:
    """Экспортирует время кадра и сходство в xlsx."""
    target = Path(file_path)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Результаты"
    sheet.append(["Время", "Сходство"])
    for frame in frames:
        sheet.append([format_timestamp(frame.timestamp), round(frame.score * 100, 2)])
    workbook.save(target)
    return target
