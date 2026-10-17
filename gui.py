import os
import re
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

from PIL import Image, ImageTk

from cugan import cugan_start
from image_utils import is_supported_image, EXTENSION_CHOICES

PREVIEW_W, PREVIEW_H = 300, 250


def get_cugan_model_scale(model_path, parent=None):
    """Return 2, 3 or 4 (from the file name, or asked from the user).
    Returns None if the user cancels the dialog."""
    match = re.search(r'up([2-4])x', os.path.basename(model_path).lower())
    if match:
        return int(match.group(1))

    while True:
        scale = simpledialog.askstring(
            "Input model scale",
            f"Could not auto-detect scale from model filename: {os.path.basename(model_path)}\n"
            "The file name might be modified, please input the scale of the loaded model "
            "(2x, 3x or 4x).\nOnly input the integer:",
            parent=parent,
        )

        if scale is None:  # Cancel / closed the dialog
            return None

        scale = scale.strip()
        if scale in ("2", "3", "4"):
            return int(scale)

        messagebox.showerror("Invalid Input",
                             "Invalid input, please type the integer (2, 3, or 4) only",
                             parent=parent)


def clean_path(text):
    """Strip whitespace and the quotes Windows adds with 'Copy as path'."""
    return text.strip().strip('"').strip("'")


class UpscalerApp:
    def __init__(self, root):
        self.root = root
        root.title("AI Image Upscaler")
        root.resizable(False, False)

        self.msg_queue = queue.Queue()
        self.preview_img = None  # keep a reference or tkinter garbage-collects it

        self.input_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.prefix_var = tk.StringVar()
        self.ext_var = tk.StringVar(value="keep original")
        self.post_var = tk.StringVar(value="off")
        self.factor_var = tk.IntVar(value=1)
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        pad = {"padx": 8, "pady": 2}
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0)

        # --- input (file or folder) ---
        ttk.Label(main, text="Open file or folder").grid(row=0, column=0, sticky="w", **pad)
        in_entry = ttk.Entry(main, textvariable=self.input_var, width=60)
        in_entry.grid(row=1, column=0, columnspan=2, sticky="we", **pad)
        in_entry.bind("<Return>", lambda e: self.update_preview())
        in_entry.bind("<FocusOut>", lambda e: self.update_preview())
        btns = ttk.Frame(main)
        btns.grid(row=1, column=2, **pad)
        ttk.Button(btns, text="File...", command=self.browse_input_file).pack(side="left")
        ttk.Button(btns, text="Folder...", command=self.browse_input_folder).pack(side="left", padx=(4, 0))

        # --- model ---
        ttk.Label(main, text="Load model (.pth)").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(main, textvariable=self.model_var, width=60).grid(row=3, column=0, columnspan=2, sticky="we", **pad)
        ttk.Button(main, text="Browse...", command=self.browse_model).grid(row=3, column=2, **pad)

        # --- preview + sliders ---
        mid = ttk.Frame(main)
        mid.grid(row=4, column=0, columnspan=3, sticky="we", pady=8)

        preview_frame = tk.Frame(mid, width=PREVIEW_W, height=PREVIEW_H, relief="solid", borderwidth=1)
        preview_frame.pack_propagate(False)
        preview_frame.pack(side="left", padx=8)
        self.preview_label = tk.Label(preview_frame, text="Preview of original input image")
        self.preview_label.pack(expand=True, fill="both")

        sliders = ttk.Frame(mid)
        sliders.pack(side="left", padx=8, fill="y")

        self.cache_scale = self._make_slider(sliders, "Cache mode", 0, 2, 1, 0, 0)
        self.tile_scale = self._make_slider(sliders, "Tile mode", 0, 5, 1, 2, 1)
        self.alpha_scale = self._make_slider(sliders, "Alpha", 0, 2, 0.1, 1, 2)

        # --- output ---
        ttk.Label(main, text="Output path").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(main, textvariable=self.output_var, width=60).grid(row=6, column=0, columnspan=2, sticky="we", **pad)
        ttk.Button(main, text="Browse...", command=self.browse_output).grid(row=6, column=2, **pad)

        ttk.Label(main, text="File prefix").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(main, textvariable=self.prefix_var, width=30).grid(row=8, column=0, sticky="w", **pad)

        ttk.Label(main, text="Extension").grid(row=9, column=0, sticky="w", **pad)
        ttk.Combobox(main, textvariable=self.ext_var, values=list(EXTENSION_CHOICES.keys()),
                     state="readonly", width=27).grid(row=10, column=0, sticky="w", **pad)

        # --- post processing ---
        post = ttk.Frame(main)
        post.grid(row=11, column=0, columnspan=3, sticky="we", pady=(10, 2))

        ttk.Label(post, text="Post processing").grid(row=0, column=0, sticky="w", padx=8)
        ttk.Label(post, text="Factor (1-8)").grid(row=0, column=1, sticky="w", padx=8)

        self.post_combo = ttk.Combobox(post, textvariable=self.post_var, state="readonly", width=14,
                                       values=["off", "bilinear", "bicubic", "lanczos"])
        self.post_combo.grid(row=1, column=0, padx=8, sticky="w")
        self.post_combo.bind("<<ComboboxSelected>>", lambda e: self._toggle_factor())

        self.factor_spin = ttk.Spinbox(post, from_=1, to=8, textvariable=self.factor_var, width=6, state="disabled")
        self.factor_spin.grid(row=1, column=1, padx=8, sticky="w")

        self.run_btn = ttk.Button(post, text="Run", command=self.on_run, width=16)
        self.run_btn.grid(row=1, column=2, padx=(120, 8))

        ttk.Label(main, textvariable=self.status_var).grid(row=12, column=0, columnspan=3, sticky="w", padx=8, pady=(8, 0))

    def _make_slider(self, parent, label, lo, hi, resolution, default, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=6)
        scale = tk.Scale(parent, from_=lo, to=hi, resolution=resolution, orient="horizontal",
                         length=260, tickinterval=1)
        scale.set(default)
        scale.grid(row=row, column=1, pady=6)
        return scale

    def _toggle_factor(self):
        state = "disabled" if self.post_var.get() == "off" else "normal"
        self.factor_spin.config(state=state)

    # ------------------------------------------------------------- browsing
    def browse_input_file(self):
        path = filedialog.askopenfilename(title="Select an image",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                                                     ("All files", "*.*")])
        if path:
            self.input_var.set(path)
            self.update_preview()

    def browse_input_folder(self):
        path = filedialog.askdirectory(title="Select a folder of images")
        if path:
            self.input_var.set(path)
            self.update_preview()

    def browse_model(self):
        path = filedialog.askopenfilename(title="Select a model", filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    # -------------------------------------------------------------- preview
    def update_preview(self):
        path = clean_path(self.input_var.get())
        img_file = None

        if os.path.isfile(path):
            img_file = path
        elif os.path.isdir(path):  # use the first readable image in the folder
            for name in sorted(os.listdir(path)):
                candidate = os.path.join(path, name)
                if is_supported_image(candidate):
                    img_file = candidate
                    break

        if not img_file or not is_supported_image(img_file):
            self.preview_img = None
            self.preview_label.config(image="", text="Preview of original input image")
            return

        try:
            with Image.open(img_file) as img:
                img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")
                img.thumbnail((PREVIEW_W - 4, PREVIEW_H - 4))
                self.preview_img = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self.preview_img, text="")
        except Exception:
            self.preview_img = None
            self.preview_label.config(image="", text="Cannot preview this image")

    # ------------------------------------------------------------------ run
    def on_run(self):
        input_path = clean_path(self.input_var.get())
        model_path = clean_path(self.model_var.get())
        output_path = clean_path(self.output_var.get())

        if not os.path.exists(input_path):
            messagebox.showerror("Invalid input", "The input file or folder does not exist.")
            return
        if not os.path.isfile(model_path):
            messagebox.showerror("Invalid model", "The model file does not exist.")
            return
        if not output_path:
            messagebox.showerror("Invalid output", "Please choose an output folder.")
            return

        prefix = self.prefix_var.get()
        if re.search(r'[\\/:*?"<>|]', prefix):
            messagebox.showerror("Invalid prefix", 'The prefix cannot contain any of these characters: \\ / : * ? " < > |')
            return

        try:
            factor = int(self.factor_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid factor", "Factor must be an integer from 1 to 8.")
            return
        if not 1 <= factor <= 8:
            messagebox.showerror("Invalid factor", "Factor must be an integer from 1 to 8.")
            return

        # Must run on the main thread because it may open dialogs
        scale = get_cugan_model_scale(model_path, parent=self.root)
        if scale is None:
            return

        post_algorithm = self.post_var.get()
        post_algorithm = None if post_algorithm == "off" else post_algorithm

        args = dict(
            input_path=input_path,
            model_path=model_path,
            scale=scale,
            tile_mode=int(self.tile_scale.get()),
            cache_mode=int(self.cache_scale.get()),
            alpha=float(self.alpha_scale.get()),
            output_path=output_path,
            prefix=prefix,
            extension=EXTENSION_CHOICES[self.ext_var.get()],
            post_algorithm=post_algorithm,
            post_factor=factor,
            progress_callback=lambda done, total: self.msg_queue.put(("progress", done, total)),
        )

        self.run_btn.config(state="disabled")
        self.status_var.set("Loading model...")
        is_folder = os.path.isdir(input_path)

        threading.Thread(target=self._worker, args=(args,), daemon=True).start()
        self.root.after(100, lambda: self._poll(is_folder))

    def _worker(self, args):
        """Runs in a background thread so the window stays responsive."""
        try:
            result = cugan_start(**args)
            self.msg_queue.put(("done", result))
        except Exception as e:
            self.msg_queue.put(("error", str(e)))

    def _poll(self, is_folder):
        finished = False
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg[0] == "progress":
                    _, done, total = msg
                    self.status_var.set(f"Processing {min(done + 1, total)} / {total} ...")
                elif msg[0] == "done":
                    self._on_done(msg[1], is_folder)
                    finished = True
                elif msg[0] == "error":
                    self.run_btn.config(state="normal")
                    self.status_var.set("Failed")
                    messagebox.showerror("Error", msg[1])
                    finished = True
        except queue.Empty:
            pass

        if not finished:
            self.root.after(100, lambda: self._poll(is_folder))

    def _on_done(self, result, is_folder):
        self.run_btn.config(state="normal")
        self.status_var.set("Done")

        if result == -1:
            messagebox.showerror("Error", "Invalid input or output path.")
            return

        if is_folder:
            self.show_summary(result)
        elif result["processed"]:
            messagebox.showinfo("Done", f"Saved to:\n{result['outputs'][0]}")
        else:
            reason = result["errors"][0][1] if result["errors"] else "The file is not a supported image."
            messagebox.showerror("Failed", reason)

    def show_summary(self, result):
        win = tk.Toplevel(self.root)
        win.title("Finished")
        win.resizable(False, False)
        win.transient(self.root)

        failed_or_skipped = result["skipped"] + len([e for e in result["errors"] if e[1] and not _is_skip_message(e)])

        frame = ttk.Frame(win, padding=15)
        frame.pack()
        ttk.Label(frame, text=f"Files processed: {result['processed']}").pack(anchor="w")
        ttk.Label(frame, text=f"Files skipped or errored: {failed_or_skipped}").pack(anchor="w", pady=(4, 0))

        if result["errors"]:
            ttk.Label(frame, text="Details:").pack(anchor="w", pady=(10, 2))
            box = tk.Text(frame, width=60, height=8, wrap="word")
            for name, msg in result["errors"]:
                box.insert("end", f"{name}: {msg}\n")
            box.config(state="disabled")
            box.pack()

        ttk.Button(frame, text="OK", command=win.destroy).pack(pady=(10, 0))


def _is_skip_message(entry):
    """Errors list also holds the 'would overwrite source' entries, which are already
    counted in result['skipped'], so don't count them twice."""
    return entry[1].startswith("output would overwrite")


def run_gui():
    root = tk.Tk()
    UpscalerApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()