# ==========================================================
# Clean Console Output - Hide Mediapipe & TF Lite native logs
# ==========================================================
import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ["KMP_AFFINITY"] = "disabled"

# Hide absl & TF python warnings
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("absl").setLevel(logging.ERROR)

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.FATAL)
except Exception:
    pass
# ==========================================================

# ==========================================================

import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import time

# --------------------------- Load Model ---------------------------
model_path = "model.h5"
model = None
if os.path.exists(model_path):
    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print("Error loading model:", e)
else:
    print(f"Error: Model file '{model_path}' not found!")

# --------------------------- Mediapipe Init ---------------------------
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# --------------------------- Theme ---------------------------
THEME = {
    "BG": "#0F172A",
    "CARD": "#111827",
    "HEADER": "#1E3A8A",
    "TEXT": "#E6EEF6",
    "SECOND": "#9AA6BF",
    "DETECT_BOX": (255, 215, 0)
}

# --------------------------- Globals ---------------------------
camera_running = False
cap = None
video_feed_frame = None
frame_lock = threading.Lock()

current_sentence = ""
detection_history = []
MAX_HISTORY = 12

last_detected_label = None
detection_stable_count = 0
MIN_STABILITY_COUNT = 5
last_append_time = 0
current_display_label = ""
DETECTION_CONF_THRESH = 0.58

# --------------------------- Helpers ---------------------------
def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    pts = [[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark]
    return pts

def pre_process_landmark(landmark_list):
    tmp = copy.deepcopy(landmark_list)
    base_x, base_y = tmp[0]
    for i in range(len(tmp)):
        tmp[i][0] -= base_x
        tmp[i][1] -= base_y
    flat = list(itertools.chain.from_iterable(tmp))
    max_val = max(map(abs, flat)) if flat else 1
    return [v / max_val for v in flat]

# --------------------------- Camera Loop ---------------------------
def start_camera_thread():
    global camera_running
    if camera_running:
        return
    camera_running = True
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

def stop_camera():
    global camera_running, cap
    camera_running = False
    if cap:
        cap.release()
        cap = None

def camera_loop():
    global cap, camera_running, video_feed_frame
    global last_detected_label, detection_stable_count, last_append_time
    global current_sentence, current_display_label, detection_history

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
            camera_running = False
            return

        with mp_hands.Hands(max_num_hands=1,
                            min_detection_confidence=0.65,
                            min_tracking_confidence=0.5) as hands:
            while camera_running:
                success, frame = cap.read()
                if not success:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                display = frame.copy()

                detected_label = None
                conf = 0.0
                bbox = None

                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(display, lm, mp_hands.HAND_CONNECTIONS)
                    landmark_list = calc_landmark_list(display, lm)
                    xs, ys = [p[0] for p in landmark_list], [p[1] for p in landmark_list]
                    bbox = (min(xs), min(ys), max(xs), max(ys))
                    data = pre_process_landmark(landmark_list)

                    if model:
                        try:
                            df = pd.DataFrame(data).transpose()
                            preds = model.predict(df, verbose=0)
                            idx = int(np.argmax(preds))
                            conf = float(np.max(preds))
                            detected_label = alphabet[idx]
                        except Exception:
                            pass

                if detected_label and conf >= DETECTION_CONF_THRESH:
                    if detected_label == last_detected_label:
                        detection_stable_count += 1
                    else:
                        last_detected_label = detected_label
                        detection_stable_count = 1

    # Smooth label display for bounding box
                    current_display_label = f"{detected_label} ({conf*100:.1f}%)"

    # After stable detection, add to sentence + history
                    if detection_stable_count >= MIN_STABILITY_COUNT:
                        now = time.time()
                        if now - last_append_time >= 2.0:
                            current_sentence += detected_label
                            last_append_time = now
                            detection_history.insert(0, f"{time.strftime('%H:%M:%S')} - {detected_label}")
                            if len(detection_history) > MAX_HISTORY:
                                detection_history.pop()
                                detection_stable_count = 0
                else:
                    detection_stable_count = 0
                    current_display_label = ""

                

                if bbox:
                    x1, y1, x2, y2 = bbox
                    pad = 10
                    cv2.rectangle(display, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), THEME["DETECT_BOX"], 3)
                    if current_display_label:
                        cv2.rectangle(display, (x1, y1 - 30), (x1 + 190, y1), THEME["DETECT_BOX"], -1)
                        cv2.putText(display, current_display_label, (x1 + 5, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                with frame_lock:
                    video_feed_frame = display.copy()

    except Exception as e:
        messagebox.showerror("Camera Error", str(e))
    finally:
        if cap:
            cap.release()
        camera_running = False

# --------------------------- UI Logic ---------------------------
def update_ui():
    global video_feed_frame, camera_running
    with frame_lock:
        frame = None if video_feed_frame is None else video_feed_frame.copy()
    if frame is not None:
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(img))
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        except Exception:
            pass

    sentence_var.set(current_sentence)
    detection_log.config(state=tk.NORMAL)
    detection_log.delete("1.0", tk.END)
    for h in detection_history:
        detection_log.insert(tk.END, h + "\n")
    detection_log.config(state=tk.DISABLED)

    if camera_running:
        root.after(20, update_ui)

def start_detection():
    start_camera_thread()
    update_ui()

def stop_detection():
    stop_camera()

def clear_sentence():
    global current_sentence
    current_sentence = ""
    sentence_var.set("")

def on_keypress(e):
    global current_sentence
    if e.keysym == "space":
        current_sentence += " "
    elif e.keysym == "BackSpace":
        current_sentence = current_sentence[:-1]
    sentence_var.set(current_sentence)

# --------------------------- Popups ---------------------------
def show_about():
    win = tk.Toplevel(root)
    win.title("About")
    win.configure(bg=THEME["CARD"])
    tk.Label(win, text="ℹ About Indian Sign Language Detector",
             bg=THEME["CARD"], fg="#E6EEF6", font=("Segoe UI", 14, "bold")).pack(pady=10)
    about_text = (
        "This project uses Artificial Intelligence and Computer Vision\n"
        "to recognize Indian Sign Language gestures in real time.\n\n"
        "🧠 Built with: TensorFlow, Mediapipe, OpenCV, Tkinter\n"
        "💡 Features:\n"
        " • Real-time gesture recognition\n"
        " • Supports A–Z and digits 1–9\n"
        " • Automatic sentence formation\n"
        " • Detection history logging\n\n"
        "Version: 2.5.0\n© 2025 Indian Sign Language Detection Project"
    )
    tk.Message(win, text=about_text, bg=THEME["CARD"], fg=THEME["SECOND"], width=460).pack(padx=20)
    tk.Button(win, text="Close", command=win.destroy, bg="#2980B9", fg="white",
              activebackground="#1F5F8B", relief="flat", padx=15, pady=5).pack(pady=10)

def show_how_to_use():
    win = tk.Toplevel(root)
    win.title("How to Use")
    win.configure(bg=THEME["CARD"])
    tk.Label(win, text="💡 How to Use Indian Sign Language Detector",
             bg=THEME["CARD"], fg="#E6EEF6", font=("Segoe UI", 14, "bold")).pack(pady=10)
    steps = (
        "1️⃣ Click **Start Detection** to open your camera.\n\n"
        "2️⃣ Keep your hand clearly visible with proper lighting.\n\n"
        "3️⃣ Hold a gesture steady for 2–3 seconds to register it.\n\n"
        "4️⃣ The detected letter will appear in the sentence area.\n\n"
        "5️⃣ Use **Space** to add space, **Backspace** to delete.\n\n"
        "6️⃣ Click **Stop Detection** or **Exit** when done.\n\n"
        "7️⃣ Your detection history will appear on the right side."
    )
    tk.Message(win, text=steps, bg=THEME["CARD"], fg=THEME["SECOND"], width=480, justify="left").pack(padx=20)
    tk.Button(win, text="Close", command=win.destroy, bg="#2980B9", fg="white",
              activebackground="#1F5F8B", relief="flat", padx=15, pady=5).pack(pady=10)

# --------------------------- Build UI ---------------------------
root = tk.Tk()
root.title("Indian Sign Language Detection System")
root.geometry("1280x720")
root.configure(bg=THEME["BG"])
root.bind("<KeyPress>", on_keypress)

header = tk.Frame(root, bg=THEME["HEADER"], pady=10)
header.pack(fill="x")
tk.Label(header, text="Indian Sign Language Detection System",
         bg=THEME["HEADER"], fg="white", font=("Segoe UI", 18, "bold")).pack()

content = tk.Frame(root, bg=THEME["BG"], padx=10, pady=10)
content.pack(fill="both", expand=True)
content.columnconfigure(0, weight=7)
content.columnconfigure(1, weight=3)

video_frame = tk.Frame(content, bg=THEME["CARD"], padx=6, pady=6)
video_frame.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)
video_label = tk.Label(video_frame, bg="black")
video_label.pack(fill="both", expand=True)

sentence_frame = tk.Frame(video_frame, bg=THEME["CARD"])
sentence_frame.pack(fill="x", pady=(8, 0))
tk.Label(sentence_frame, text="Sentence:", bg=THEME["CARD"], fg="white").pack(side="left", padx=5)
sentence_var = tk.StringVar()
tk.Label(sentence_frame, textvariable=sentence_var, bg=THEME["CARD"], fg="#22C55E",
         font=("Segoe UI", 14, "bold")).pack(side="left", padx=5)
tk.Button(sentence_frame, text="Clear", command=clear_sentence,
          bg="#C0392B", fg="white").pack(side="right", padx=6)

right = tk.Frame(content, bg=THEME["BG"], padx=8)
right.grid(row=0, column=1, sticky="nsew")
controls = tk.Frame(right, bg=THEME["CARD"], padx=12, pady=12)
controls.pack(fill="x", pady=(0, 8))

tk.Label(controls, text="Controls", bg=THEME["CARD"], fg="white",
         font=("Segoe UI", 13, "bold")).pack(anchor="w")

# ----------- Soft Professional Buttons -----------
def create_button(parent, text, command, bg_color, hover_color):
    btn = tk.Label(parent, text=text, bg=bg_color, fg="white",
                   font=("Segoe UI", 11, "bold"), padx=12, pady=8, cursor="hand2", relief="flat", bd=0)
    btn.pack(fill="x", pady=4)
    btn.bind("<Button-1>", lambda e: command())
    btn.bind("<Enter>", lambda e: btn.config(bg=hover_color))
    btn.bind("<Leave>", lambda e: btn.config(bg=bg_color))
    return btn

create_button(controls, "▶ Start Detection", start_detection, "#2E8B57", "#24704A")
create_button(controls, "⏹ Stop Detection", stop_detection, "#C0392B", "#992D22")
create_button(controls, "💡 How to Use", show_how_to_use, "#2980B9", "#1F5F8B")
create_button(controls, "ℹ About", show_about, "#8E44AD", "#6D3393")
create_button(controls, "🚪 Exit Application", lambda: (stop_camera(), root.destroy()), "#E67E22", "#CA6F1E")

history = tk.Frame(right, bg=THEME["CARD"], padx=10, pady=10)
history.pack(fill="both", expand=True)
tk.Label(history, text="Detection History", bg=THEME["CARD"],
         fg="white", font=("Segoe UI", 13, "bold")).pack(anchor="w")
detection_log = tk.Text(history, bg="#0B0C0F", fg="white", height=15)
detection_log.pack(fill="both", expand=True)
detection_log.config(state=tk.DISABLED)

footer = tk.Frame(root, bg=THEME["CARD"], pady=8)
footer.pack(fill="x")
tk.Label(footer, text="© 2025 Indian Sign Language Detection Project",
         bg=THEME["CARD"], fg=THEME["SECOND"], font=("Segoe UI", 9)).pack()

root.protocol("WM_DELETE_WINDOW", lambda: (stop_camera(), root.destroy()))
root.mainloop()
