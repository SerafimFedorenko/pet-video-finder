"""Главное окно приложения. Содержит только UI и не работает с моделями напрямую."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.messagebox import askyesno, showinfo, showerror

from PIL import Image, ImageTk

from config import CAT_CLASS_ID, DOG_CLASS_ID, MODE_IMAGE, MODE_TEXT
from controllers import (
    AnalysisRequest,
    AnalysisResult,
    FrameResult,
    VideoAnalysisController,
    export_results,
    filter_frames,
    save_best_crops,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Русские названия животных -> индексы классов COCO.
PETS = {"Кошки": CAT_CLASS_ID, "Собаки": DOG_CLASS_ID}


def validate_digits(value: str) -> bool:
    """Валидатор для полей ввода: только цифры или пустая строка."""
    return value.isdigit() or value == ""


class MainWindow:
    def __init__(
        self, window: tk.Tk, controller: VideoAnalysisController | None = None
    ):
        self.window = window
        # Контроллер можно подменить в тестах, не загружая Keras и YOLO.
        self.controller = controller or VideoAnalysisController()

        self.frames: list[FrameResult] = []
        self._all_frames: list[FrameResult] = []
        self.current_index = 0
        self.current_image_pil: Image.Image | None = None
        self._processing = False

        self.window.title("Pet Video Finder — поиск животных на видео")
        icon_path = PROJECT_ROOT / "pet.ico"
        if icon_path.exists():
            try:
                self.window.iconbitmap(str(icon_path))
            except tk.TclError:
                pass
        self.window.configure(bg="#2e2e2e")
        try:
            self.window.state("zoomed")
        except tk.TclError:
            pass

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        self.window.minsize(
            width=min(screen_width, 1024), height=min(screen_height, 700)
        )
        self.window.maxsize(width=screen_width, height=screen_height)
        self.window.resizable(False, False)
        self.frame_width = 800
        self.frame_height = 600

        self.vcmd = self.window.register(validate_digits)

        label_font = ("Helvetica", 12)
        button_font = ("Helvetica", 12, "bold")
        bg_color = "#2e2e2e"
        text_color = "#FFFFFF"
        button_color = "#3e3e3e"
        entry_bg_color = "#4e4e4e"
        entry_fg_color = "#FFFFFF"

        self.canvas = tk.Canvas(
            window, width=self.frame_width, height=self.frame_height, bg="white"
        )
        self.canvas.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        self.selected_image_canvas = tk.Canvas(
            window,
            width=self.frame_width // 4,
            height=self.frame_height // 4,
            bg="white",
        )
        self.selected_image_canvas.grid(row=0, column=1, padx=20, pady=20, sticky="nw")

        button_frame = tk.Frame(window, bg=bg_color)
        button_frame.grid(row=1, column=0, padx=20, pady=5)
        prev_button = tk.Button(
            button_frame,
            text="Назад",
            command=self.show_previous_image,
            bg=button_color,
            fg=text_color,
            font=button_font,
        )
        prev_button.pack(side=tk.LEFT, padx=10)
        next_button = tk.Button(
            button_frame,
            text="Вперед",
            command=self.show_next_image,
            bg=button_color,
            fg=text_color,
            font=button_font,
        )
        next_button.pack(side=tk.LEFT, padx=10)

        self.image_counter = tk.Label(
            window, text="Кадр 0 из 0", bg=bg_color, fg=text_color, font=label_font
        )
        self.image_counter.grid(row=2, column=0, pady=5)

        control_frame = tk.Frame(window, bg=bg_color)
        control_frame.grid(row=0, column=2, rowspan=7, padx=20, pady=20, sticky="ne")

        self.select_pet_combobox = ttk.Combobox(
            control_frame, values=list(PETS), state="readonly", font=label_font
        )
        self.select_pet_combobox.set("Кошки")
        self.select_pet_combobox.pack(fill=tk.X, pady=(0, 10))

        label = tk.Label(
            control_frame,
            text="Интервал между кадрами (в секундах):",
            bg=bg_color,
            fg=text_color,
            font=label_font,
        )
        label.pack(fill=tk.X)
        self.interval_entry = tk.Entry(
            control_frame,
            validate="all",
            validatecommand=(self.vcmd, "%P"),
            bg=entry_bg_color,
            fg=entry_fg_color,
            font=label_font,
            width=3,
        )
        self.interval_entry.pack(fill=tk.X, pady=(0, 10))

        self.select_video_button = tk.Button(
            control_frame,
            text="Выбрать видео",
            command=self.select_video,
            bg=button_color,
            fg=text_color,
            font=button_font,
        )
        self.select_video_button.pack(fill=tk.X, pady=(0, 10))

        select_image_button = tk.Button(
            control_frame,
            text="Выбрать изображение",
            command=self.select_image,
            bg=button_color,
            fg=text_color,
            font=button_font,
        )
        select_image_button.pack(fill=tk.X, pady=(0, 10))

        self.current_image_path_label = tk.Label(
            control_frame,
            text="Текущее изображение:",
            bg=bg_color,
            fg=text_color,
            font=label_font,
        )
        self.current_image_path_label.pack(fill=tk.X)
        self.current_image_path = tk.Entry(
            control_frame,
            bg=entry_bg_color,
            fg=entry_fg_color,
            font=label_font,
            width=50,
        )
        self.current_image_path.pack(fill=tk.X, pady=(0, 10))

        label_description = tk.Label(
            control_frame,
            text="Описание:",
            bg=bg_color,
            fg=text_color,
            font=label_font,
        )
        label_description.pack(fill=tk.X)
        self.description_entry = tk.Text(
            control_frame,
            bg=entry_bg_color,
            fg=entry_fg_color,
            font=label_font,
            width=40,
            height=4,
        )
        self.description_entry.pack(fill=tk.X, pady=(0, 10))

        self.selected_state = tk.StringVar(value="text_option")

        self.image_option = tk.Radiobutton(
            control_frame,
            text="Сравнивать с изображением",
            variable=self.selected_state,
            value="image_option",
            bg=bg_color,
            fg=text_color,
            font=label_font,
            selectcolor=button_color,
        )
        self.text_option = tk.Radiobutton(
            control_frame,
            text="Сравнивать с описанием",
            variable=self.selected_state,
            value="text_option",
            bg=bg_color,
            fg=text_color,
            font=label_font,
            selectcolor=button_color,
        )
        self.image_option.pack(fill=tk.X, pady=(0, 5))
        self.text_option.pack(fill=tk.X, pady=(0, 10))

        self.probality_label = tk.Label(
            control_frame,
            text="Соответствие описанию / изображению: ",
            bg=bg_color,
            fg=text_color,
            font=label_font,
        )
        self.probality_label.pack(pady=(10, 10))

        save_button = tk.Button(
            control_frame,
            text="Сохранить изображение",
            command=self.save_image,
            bg=button_color,
            fg=text_color,
            font=button_font,
        )
        save_button.pack(fill=tk.X, pady=(0, 10))

        self.progress = ttk.Progressbar(
            control_frame, orient=tk.HORIZONTAL, length=200, mode="determinate"
        )
        self.progress.pack(pady=10)

        self.filter_frame = tk.Frame(control_frame, bg=bg_color)
        self.filter_frame.pack(fill=tk.X, pady=(10, 10))

        self.probability_label = tk.Label(
            self.filter_frame,
            text="Фильтр по вероятности (%)",
            bg=bg_color,
            fg=text_color,
            font=label_font,
        )
        self.probability_label.pack(fill=tk.X)

        self.probability_entry = tk.Entry(
            self.filter_frame,
            validate="all",
            validatecommand=(self.vcmd, "%P"),
            bg=entry_bg_color,
            fg=entry_fg_color,
            font=label_font,
            width=3,
        )
        self.probability_entry.pack(fill=tk.X, pady=(0, 10))

        apply_filter_button = tk.Button(
            self.filter_frame,
            text="Применить фильтр",
            command=self.apply_filter,
            bg=button_color,
            fg=text_color,
            font=button_font,
        )
        apply_filter_button.pack(fill=tk.X, pady=(0, 10))

        save_all_button = tk.Button(
            control_frame,
            text="Сохранить все подходящие изображения",
            command=self.save_all_images,
            bg=button_color,
            fg=text_color,
            font=button_font,
        )
        save_all_button.pack(fill=tk.X, pady=(0, 10))

        export_results_button = tk.Button(
            control_frame,
            text="Экспорт результатов в Excel",
            command=self.export_results,
            bg=button_color,
            fg=text_color,
            font=button_font,
        )
        export_results_button.pack(fill=tk.X, pady=(0, 10))

        self.window.bind("<Left>", lambda _event: self.show_previous_image())
        self.window.bind("<Right>", lambda _event: self.show_next_image())

    # ---------- вспомогательные методы ----------

    def _ui(self, func, *args) -> None:
        """Выполняет действие в главном потоке Tkinter."""
        self.window.after(0, lambda: func(*args))

    def _set_progress(self, value: int) -> None:
        def update() -> None:
            self.progress["maximum"] = 100
            self.progress["value"] = max(0, min(100, value))

        self._ui(update)

    def _interval(self) -> int:
        try:
            return max(1, int(self.interval_entry.get()))
        except ValueError:
            return 1

    def _index(self) -> int:
        return PETS.get(self.select_pet_combobox.get(), CAT_CLASS_ID)

    def _reference_image(self) -> Image.Image | None:
        """Эталонное изображение из поля пути (файл можно ввести вручную)."""
        path = self.current_image_path.get().strip()
        if not path:
            return None
        if (
            self.current_image_pil is None
            or getattr(self, "_reference_path", None) != path
        ):
            try:
                self.current_image_pil = Image.open(path).convert("RGB")
                self._reference_path = path
            except Exception:  # noqa: BLE001
                return None
        return self.current_image_pil

    # ---------- выбор исходных данных ----------

    def select_image(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[("Изображение", "*.jpg;*.jpeg;*.png"), ("Все файлы", "*.*")]
        )
        if not file_path:
            return
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            showerror("Ошибка", f"Не удалось открыть изображение:\n{exc}")
            return

        self.current_image_pil = image
        self._reference_path = file_path
        preview = image.copy()
        preview.thumbnail(
            (self.frame_width // 4, self.frame_height // 4), Image.Resampling.LANCZOS
        )
        photo = ImageTk.PhotoImage(preview)
        self.selected_image_canvas.delete("all")
        self.selected_image_canvas.create_image(
            self.frame_width // 8,
            self.frame_height // 8,
            image=photo,
            anchor="center",
        )
        self.selected_image_canvas.image = photo
        self.current_image_path.delete(0, tk.END)
        self.current_image_path.insert(0, file_path)

    def select_video(self) -> None:
        if self._processing:
            showinfo("Обработка", "Дождитесь завершения текущей обработки видео.")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("Видео", "*.mp4;*.avi;*.mkv;*.mov"), ("Все файлы", "*.*")]
        )
        if not file_path:
            return

        mode = (
            MODE_IMAGE
            if self.selected_state.get() == "image_option"
            else MODE_TEXT
        )
        request = AnalysisRequest(
            class_index=self._index(),
            interval=self._interval(),
            mode=mode,
            description=self.description_entry.get("1.0", "end-1c").strip(),
            reference_image=self._reference_image(),
        )
        try:
            request.validate()
        except ValueError as exc:
            showinfo("Недостаточно данных", str(exc))
            return

        try:
            estimated = self.controller.estimate_time(file_path, request)
        except Exception as exc:  # noqa: BLE001
            showerror("Ошибка", f"Не удалось прочитать видео:\n{exc}")
            return

        minutes, seconds = divmod(int(estimated), 60)
        result = askyesno(
            title="Подтверждение операции",
            message=(
                f"Расчётное время выполнения: {minutes} мин {seconds} с. Продолжить?"
            ),
        )
        if not result:
            showinfo("Результат", "Операция отменена")
            return

        self._processing = True
        self.select_video_button.config(state=tk.DISABLED)
        self._clear_results()
        self._set_progress(0)
        threading.Thread(
            target=self._process_video,
            args=(file_path, request),
            daemon=True,
        ).start()

    # ---------- обработка видео ----------

    def _process_video(self, file_path: str, request: AnalysisRequest) -> None:
        try:
            result = self.controller.analyze(
                file_path, request, progress_cb=self._on_progress
            )
            if not result.frames:
                self._ui(
                    showinfo, "Результат обработки", "Подходящих кадров не найдено"
                )
            else:
                self._ui(self._show_results, result)
        except Exception as exc:  # noqa: BLE001
            self._ui(showerror, "Ошибка обработки", str(exc))
        finally:
            self._ui(self._finish_processing)

    def _on_progress(self, percent: int) -> None:
        self._set_progress(percent)

    def _finish_processing(self) -> None:
        self._processing = False
        self.select_video_button.config(state=tk.NORMAL)
        self._set_progress(100)

    def _clear_results(self) -> None:
        self.frames = []
        self._all_frames = []
        self.current_index = 0
        self.canvas.delete("all")
        self.image_counter.config(text="Кадр 0 из 0")
        self.probality_label.config(text="Соответствие описанию / изображению: ")

    def _show_results(self, result: AnalysisResult) -> None:
        self._all_frames = list(result.frames)
        self.frames = list(self._all_frames)
        self.current_index = 0
        self.show_image()

    # ---------- отображение ----------

    def show_image(self) -> None:
        if not self.frames:
            return
        self.current_index %= len(self.frames)
        frame = self.frames[self.current_index]
        image = frame.image.copy()
        image.thumbnail(
            (self.frame_width, self.frame_height), Image.Resampling.LANCZOS
        )
        photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.frame_width // 2,
            self.frame_height // 2,
            image=photo,
            anchor="center",
        )
        self.canvas.image = photo
        self.image_counter.config(
            text=f"Кадр {self.current_index + 1} из {len(self.frames)}"
        )
        self.probality_label.config(
            text=(
                "Соответствие описанию / изображению: "
                f"{round(frame.score * 100, 2)}%"
            )
        )

    def show_previous_image(self) -> None:
        if not self.frames:
            return
        self.current_index = (self.current_index - 1) % len(self.frames)
        self.show_image()

    def show_next_image(self) -> None:
        if not self.frames:
            return
        self.current_index = (self.current_index + 1) % len(self.frames)
        self.show_image()

    # ---------- сохранение и фильтрация ----------

    def save_image(self) -> None:
        if not self.frames:
            showinfo("Сохранение изображения", "Сначала обработайте видео.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
        )
        if file_path:
            self.frames[self.current_index].image.save(file_path)
            showinfo("Сохранение изображения", "Изображение сохранено успешно.")

    def apply_filter(self) -> None:
        if not self._all_frames:
            showinfo("Фильтр", "Сначала обработайте видео.")
            return
        try:
            threshold = float(self.probability_entry.get() or 0)
        except ValueError:
            threshold = 0.0

        selected = filter_frames(self._all_frames, threshold)
        if not selected:
            showinfo(
                "Результат фильтрации",
                "Нет изображений, соответствующих заданному порогу.",
            )
            return

        self.frames = selected
        self.current_index = 0
        self.show_image()

    def save_all_images(self) -> None:
        if not self.frames:
            showinfo("Сохранение изображений", "Нет изображений для сохранения.")
            return
        directory = filedialog.askdirectory()
        if not directory:
            return
        saved = save_best_crops(self.frames, directory)
        showinfo("Сохранение изображений", f"Сохранено файлов: {saved}.")

    def export_results(self) -> None:
        if not self.frames:
            showinfo("Экспорт результатов", "Нет результатов для экспорта.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")]
        )
        if not file_path:
            return
        try:
            export_results(self.frames, file_path)
        except Exception as exc:  # noqa: BLE001
            showerror("Ошибка экспорта", str(exc))
            return
        showinfo(
            "Экспорт результатов", "Результаты успешно экспортированы в Excel файл."
        )
