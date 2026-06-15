import customtkinter as ctk
import tkinter as tk
import cv2
import threading
import time
import pyttsx3
from PIL import Image, ImageTk
from datetime import datetime
import queue
from inference_sdk import InferenceHTTPClient

# ─── CONFIG ───────────────────────────────────────────────────────────────────
API_KEY        = "0muhNzus6VvhbWfFeNp6"
MODEL_LETTERS  = "american-sign-language-letters-gxpdm/4"
MODEL_WORDS    = "sign-language-classifier-pjgqf/1"
CONFIDENCE     = 0.50
FRAME_INTERVAL = 0.3
HOLD_THRESHOLD = 4

# ─── COLORS ───────────────────────────────────────────────────────────────────
BLUE_DARK    = "#0C447C"
BLUE_MID     = "#185FA5"
BLUE_LIGHT   = "#B5D4F4"
BLUE_XLIGHT  = "#E6F1FB"
AMBER_MID    = "#EF9F27"
AMBER_LIGHT  = "#FAC775"
AMBER_XLIGHT = "#FAEEDA"
WHITE        = "#FFFFFF"
GRAY_DARK    = "#2C2C2A"
GRAY_MID     = "#5F5E5A"
GRAY_LIGHT   = "#D3D1C7"
GRAY_XLIGHT  = "#F1EFE8"

# ─── ROBOFLOW CLIENT ──────────────────────────────────────────────────────────
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# ─── APPEARANCE ───────────────────────────────────────────────────────────────
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ─── TTS ──────────────────────────────────────────────────────────────────────
_tts_engine = pyttsx3.init()
_tts_engine.setProperty("rate", 150)

def speak(text: str):
    def _run():
        _tts_engine.say(text)
        _tts_engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()

# ─── INFERENCE ────────────────────────────────────────────────────────────────
def predict_frame(image_path: str, model_id: str):
    try:
        result = CLIENT.infer(image_path, model_id=model_id)
        predictions = result.get("predictions", [])
        if not predictions:
            return "", 0.0, []
        best = max(predictions, key=lambda p: p["confidence"])
        if best["confidence"] >= CONFIDENCE:
            return best["class"], best["confidence"], predictions
        return "", 0.0, predictions
    except Exception as e:
        print(f"Inference error: {e}")
        return "", 0.0, []


def draw_boxes(frame, predictions, orig_w, orig_h):
    fh, fw = frame.shape[:2]
    scale_x = fw / orig_w
    scale_y = fh / orig_h
    for pred in predictions:
        if pred["confidence"] < CONFIDENCE:
            continue
        cx = pred["x"] * scale_x
        cy = pred["y"] * scale_y
        w  = pred["width"]  * scale_x
        h  = pred["height"] * scale_y
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(cx + w/2), int(cy + h/2)
        label = f"{pred['class']}  {int(pred['confidence']*100)}%"
        box_color = (24, 95, 165)     # BLUE_MID in BGR
        text_bg   = (239, 159, 39)    # AMBER_MID in BGR
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), text_bg, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (12, 68, 124), 2)
    return frame


# ─── 3D BUTTON ────────────────────────────────────────────────────────────────
class Button3D(tk.Button):
    """A button with a 3D pressed effect using relief and dynamic color shifts."""
    def __init__(self, parent, label="", bg=BLUE_MID, fg=WHITE,
                 shadow="#0C447C", cmd=None, w=120, **kwargs):
        super().__init__(
            parent, text=label, bg=bg, fg=fg,
            activebackground=shadow, activeforeground=WHITE,
            font=("Segoe UI", 11, "bold"),
            relief="raised", bd=3,
            cursor="hand2", command=cmd,
            width=w, padx=10, pady=6,
            highlightthickness=0,
            **kwargs
        )
        self._bg     = bg
        self._shadow = shadow
        self.bind("<ButtonPress-1>",   self._press)
        self.bind("<ButtonRelease-1>", self._release)

    def _press(self, e):
        self.config(relief="sunken", bg=self._shadow)

    def _release(self, e):
        self.config(relief="raised", bg=self._bg)


# ─── MAIN APP ─────────────────────────────────────────────────────────────────
class ASLInterpreterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ASL Sign Language Interpreter")
        self.geometry("1200x740")
        self.resizable(True, True)
        self.configure(fg_color=BLUE_XLIGHT)

        self.cap             = None
        self.running         = False
        self.flip            = True
        self.mode            = tk.StringVar(value="Letters")
        self.sentence_words  = []
        self.last_sign       = ""
        self.sign_hold_count = 0
        self.frame_queue     = queue.Queue(maxsize=1)
        self._canvas_img     = None
        self._pred_lock      = threading.Lock()
        self._latest_preds   = []
        self._orig_w         = 640
        self._orig_h         = 480

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_canvas()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        # ── TOP BAR ──
        topbar = tk.Frame(self, bg=BLUE_DARK, height=62)
        topbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        topbar.grid_columnconfigure(1, weight=1)
        topbar.grid_propagate(False)

        # Logo area
        logo_frame = tk.Frame(topbar, bg=BLUE_DARK)
        logo_frame.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        tk.Label(logo_frame, text="ASL", bg=AMBER_MID, fg=BLUE_DARK,
                 font=("Segoe UI", 13, "bold"), padx=8, pady=2,
                 relief="raised", bd=2).pack(side="left", padx=(0, 8))
        tk.Label(logo_frame, text="Sign Language Interpreter",
                 bg=BLUE_DARK, fg=WHITE,
                 font=("Segoe UI", 15, "bold")).pack(side="left")

        # Mode toggle
        mode_frame = tk.Frame(topbar, bg=BLUE_DARK)
        mode_frame.grid(row=0, column=1, pady=10)
        tk.Label(mode_frame, text="Detection Mode:", bg=BLUE_DARK, fg=BLUE_LIGHT,
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 8))
        for val, txt in [("Letters", "Letters (A–Z)"), ("Words", "Words & Phrases")]:
            tk.Radiobutton(mode_frame, text=txt, variable=self.mode, value=val,
                           bg=BLUE_DARK, fg=WHITE, selectcolor=BLUE_MID,
                           activebackground=BLUE_DARK, activeforeground=AMBER_LIGHT,
                           font=("Segoe UI", 10), bd=0).pack(side="left", padx=6)

        self.status_dot = tk.Label(topbar, text="  ●  Stopped",
                                    bg=BLUE_DARK, fg=GRAY_LIGHT,
                                    font=("Segoe UI", 11, "bold"))
        self.status_dot.grid(row=0, column=2, padx=20)

        # ── LEFT: Camera ──
        left_outer = tk.Frame(self, bg=BLUE_XLIGHT)
        left_outer.grid(row=1, column=0, padx=(16, 8), pady=16, sticky="nsew")
        left_outer.grid_rowconfigure(1, weight=1)
        left_outer.grid_columnconfigure(0, weight=1)

        # Camera card
        cam_card = tk.Frame(left_outer, bg=WHITE, relief="raised", bd=2)
        cam_card.grid(row=0, column=0, sticky="nsew", rowspan=2)
        cam_card.grid_rowconfigure(1, weight=1)
        cam_card.grid_columnconfigure(0, weight=1)

        tk.Label(cam_card, text="  Camera Feed", bg=BLUE_MID, fg=WHITE,
                 font=("Segoe UI", 12, "bold"), anchor="w", pady=8
                 ).grid(row=0, column=0, sticky="ew")

        self.canvas = tk.Canvas(cam_card, bg="#1a2744", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.canvas.create_text(300, 200,
                                text="Camera will appear here\nafter pressing Start",
                                fill="#4a6fa5", font=("Segoe UI", 13), tags="placeholder")

        # Control bar
        ctrl_bar = tk.Frame(cam_card, bg=BLUE_XLIGHT, pady=8)
        ctrl_bar.grid(row=2, column=0, sticky="ew", padx=4, pady=(4, 8))
        ctrl_bar.grid_columnconfigure((0, 1, 2), weight=1)

        self.start_btn = Button3D(ctrl_bar, "▶  Start",
                                   bg=BLUE_MID, shadow=BLUE_DARK,
                                   cmd=self._start_camera, w=10)
        self.start_btn.grid(row=0, column=0, padx=6)

        self.stop_btn = Button3D(ctrl_bar, "■  Stop",
                                  bg=GRAY_MID, shadow=GRAY_DARK,
                                  cmd=self._stop_camera, w=10)
        self.stop_btn.grid(row=0, column=1, padx=6)
        self.stop_btn.config(state="disabled")

        Button3D(ctrl_bar, "⟳  Flip",
                 bg=AMBER_MID, fg=BLUE_DARK, shadow="#BA7517",
                 cmd=lambda: setattr(self, "flip", not self.flip), w=10
                 ).grid(row=0, column=2, padx=6)

        # ── RIGHT PANEL ──
        right = tk.Frame(self, bg=BLUE_XLIGHT)
        right.grid(row=1, column=1, padx=(8, 16), pady=16, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=1)

        # Detected sign card
        sign_card = tk.Frame(right, bg=WHITE, relief="raised", bd=2)
        sign_card.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        sign_card.grid_columnconfigure(0, weight=1)

        tk.Label(sign_card, text="  Detected Sign", bg=BLUE_MID, fg=WHITE,
                 font=("Segoe UI", 12, "bold"), anchor="w", pady=8
                 ).grid(row=0, column=0, sticky="ew")

        self.sign_label = tk.Label(sign_card, text="—",
                                    bg=WHITE, fg=BLUE_MID,
                                    font=("Segoe UI", 62, "bold"))
        self.sign_label.grid(row=1, column=0, pady=(10, 2))

        self.mode_tag = tk.Label(sign_card, text="",
                                  bg=WHITE, fg=GRAY_MID,
                                  font=("Segoe UI", 10))
        self.mode_tag.grid(row=2, column=0)

        # Progress bar (canvas-based for custom color)
        self.conf_canvas = tk.Canvas(sign_card, height=10, bg=GRAY_LIGHT,
                                      highlightthickness=0)
        self.conf_canvas.grid(row=3, column=0, sticky="ew", padx=20, pady=6)
        self.conf_fill = self.conf_canvas.create_rectangle(0, 0, 0, 10,
                                                            fill=AMBER_MID, outline="")

        self.conf_label = tk.Label(sign_card, text="Confidence: —",
                                    bg=WHITE, fg=GRAY_MID,
                                    font=("Segoe UI", 10))
        self.conf_label.grid(row=4, column=0, pady=(0, 12))

        # Sentence builder card
        sent_card = tk.Frame(right, bg=WHITE, relief="raised", bd=2)
        sent_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        sent_card.grid_columnconfigure(0, weight=1)

        tk.Label(sent_card, text="  Sentence Builder", bg=BLUE_MID, fg=WHITE,
                 font=("Segoe UI", 12, "bold"), anchor="w", pady=8
                 ).grid(row=0, column=0, sticky="ew")

        self.sentence_box = tk.Text(sent_card, height=3, font=("Segoe UI", 14),
                                     bg=BLUE_XLIGHT, fg=BLUE_DARK,
                                     relief="flat", bd=0, wrap="word",
                                     state="disabled")
        self.sentence_box.grid(row=1, column=0, sticky="ew", padx=10, pady=8)

        sent_btns = tk.Frame(sent_card, bg=WHITE, pady=8)
        sent_btns.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
        sent_btns.grid_columnconfigure((0, 1, 2), weight=1)

        Button3D(sent_btns, "Speak",
                 bg=BLUE_MID, shadow=BLUE_DARK,
                 cmd=self._speak_sentence, w=8
                 ).grid(row=0, column=0, padx=4)
        Button3D(sent_btns, "Undo",
                 bg=GRAY_MID, shadow=GRAY_DARK,
                 cmd=self._undo_word, w=8
                 ).grid(row=0, column=1, padx=4)
        Button3D(sent_btns, "Clear",
                 bg=AMBER_MID, fg=BLUE_DARK, shadow="#BA7517",
                 cmd=self._clear_sentence, w=8
                 ).grid(row=0, column=2, padx=4)

        # History card
        hist_card = tk.Frame(right, bg=WHITE, relief="raised", bd=2)
        hist_card.grid(row=2, column=0, sticky="nsew")
        hist_card.grid_columnconfigure(0, weight=1)
        hist_card.grid_rowconfigure(1, weight=1)

        tk.Label(hist_card, text="  History", bg=BLUE_MID, fg=WHITE,
                 font=("Segoe UI", 12, "bold"), anchor="w", pady=8
                 ).grid(row=0, column=0, sticky="ew")

        self.history_box = tk.Text(hist_card, height=6, font=("Segoe UI", 11),
                                    bg=BLUE_XLIGHT, fg=GRAY_MID,
                                    relief="flat", bd=0, state="disabled")
        self.history_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=8)

        # Footer
        tk.Label(self,
                 text="Powered by Roboflow  •  ASL Capstone Project",
                 bg=BLUE_XLIGHT, fg=GRAY_MID,
                 font=("Segoe UI", 10)).grid(row=2, column=0, columnspan=2, pady=(0, 6))

    # ── CONFIDENCE BAR ────────────────────────────────────────────────────────
    def _update_conf_bar(self, conf: float):
        self.conf_canvas.update_idletasks()
        w = self.conf_canvas.winfo_width()
        fill_w = int(w * conf)
        self.conf_canvas.coords(self.conf_fill, 0, 0, fill_w, 10)

    # ── CANVAS UPDATER ────────────────────────────────────────────────────────
    def _update_canvas(self):
        try:
            frame = self.frame_queue.get_nowait()
            with self._pred_lock:
                preds  = list(self._latest_preds)
                orig_w = self._orig_w
                orig_h = self._orig_h
            if preds:
                frame = draw_boxes(frame, preds, orig_w, orig_h)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 1 and ch > 1:
                # Crop to square using the shorter side, then resize — no warping
                iw, ih = img.size
                side = min(iw, ih)
                left = (iw - side) // 2
                top  = (ih - side) // 2
                img  = img.crop((left, top, left + side, top + side))
                # Fit into canvas keeping square aspect ratio
                size = min(cw, ch)
                img  = img.resize((size, size), Image.LANCZOS)
                # Center on canvas
                ox = (cw - size) // 2
                oy = (ch - size) // 2
            else:
                ox, oy = 0, 0
            self._canvas_img = ImageTk.PhotoImage(image=img)
            self.canvas.delete("all")
            self.canvas.configure(bg="#1a2744")
            self.canvas.create_image(ox, oy, anchor="nw", image=self._canvas_img)
        except queue.Empty:
            pass
        finally:
            self.after(30, self._update_canvas)

    # ── CAMERA ────────────────────────────────────────────────────────────────
    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._set_status("●  No camera found", "#E24B4A")
            return
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._set_status("●  Running", "#3B6D11")
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.canvas.delete("all")
        self.canvas.create_text(300, 200,
                                text="Camera stopped.\nPress Start to resume.",
                                fill="#4a6fa5", font=("Segoe UI", 13))
        self._set_status("●  Stopped", GRAY_MID)
        self._update_sign("—", 0.0, "")

    def _camera_loop(self):
        last_api_call = 0
        tmp_path = "tmp_frame.jpg"
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.flip:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass
            now = time.time()
            if now - last_api_call >= FRAME_INTERVAL:
                last_api_call = now
                cv2.imwrite(tmp_path, frame)
                mode  = self.mode.get()
                model = MODEL_LETTERS if mode == "Letters" else MODEL_WORDS
                threading.Thread(target=self._run_inference,
                                 args=(tmp_path, model, mode, w, h),
                                 daemon=True).start()
            time.sleep(0.03)

    def _run_inference(self, path, model, mode, w, h):
        sign, conf, preds = predict_frame(path, model)
        with self._pred_lock:
            self._latest_preds = preds
            self._orig_w = w
            self._orig_h = h
        self.after(0, self._handle_prediction, sign, conf, mode)

    # ── PREDICTION LOGIC ──────────────────────────────────────────────────────
    def _handle_prediction(self, sign: str, conf: float, mode: str):
        self._update_sign(sign if sign else "—", conf, mode if sign else "")
        if not sign:
            self.sign_hold_count = 0
            self.last_sign = ""
            return
        if sign == self.last_sign:
            self.sign_hold_count += 1
        else:
            self.sign_hold_count = 1
            self.last_sign = sign
        if self.sign_hold_count == HOLD_THRESHOLD:
            self.sentence_words.append(sign)
            self._refresh_sentence()
            self.sign_hold_count = 0

    def _update_sign(self, sign: str, conf: float, mode: str):
        self.sign_label.config(text=sign)
        self.mode_tag.config(text=f"[ {mode} ]" if mode else "")
        self._update_conf_bar(conf)
        self.conf_label.config(
            text=f"Confidence: {int(conf * 100)}%" if conf > 0 else "Confidence: —"
        )

    # ── SENTENCE BUILDER ──────────────────────────────────────────────────────
    def _refresh_sentence(self):
        text = " ".join(self.sentence_words)
        self.sentence_box.config(state="normal")
        self.sentence_box.delete("1.0", "end")
        self.sentence_box.insert("1.0", text)
        self.sentence_box.config(state="disabled")

    def _speak_sentence(self):
        text = " ".join(self.sentence_words)
        if text:
            speak(text)
            self._add_history(text)

    def _undo_word(self):
        if self.sentence_words:
            self.sentence_words.pop()
            self._refresh_sentence()

    def _clear_sentence(self):
        if self.sentence_words:
            self._add_history(" ".join(self.sentence_words))
        self.sentence_words = []
        self._refresh_sentence()

    def _add_history(self, text: str):
        ts = datetime.now().strftime("%H:%M")
        entry = f"[{ts}]  {text}\n"
        self.history_box.config(state="normal")
        self.history_box.insert("end", entry)
        self.history_box.see("end")
        self.history_box.config(state="disabled")

    def _set_status(self, text: str, color: str):
        self.status_dot.config(text=f"  {text}", fg=color)

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = ASLInterpreterApp()
    app.mainloop()