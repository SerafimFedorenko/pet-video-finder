"""Точка входа приложения «Поиск животных на видео»."""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter.messagebox import showerror

# Меньше служебных сообщений TensorFlow в консоли.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Transformers не должен использовать TensorFlow с Keras 3 — CLIP работает на PyTorch.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from ui import MainWindow


def main() -> int:
    window = tk.Tk()
    try:
        MainWindow(window)
    except Exception as exc:  # noqa: BLE001
        window.withdraw()
        showerror(
            "Ошибка запуска",
            "Не удалось загрузить модели или инициализировать окно:\n\n"
            f"{exc}",
        )
        window.destroy()
        raise
    window.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
