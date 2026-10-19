"""Real-time sign recognition from the webcam.

Usage:
    python realtime.py              # glosses only
    python realtime.py --grammar    # also turn the gloss sequence into an English sentence
    python realtime.py --debug      # also log every prediction to debug_probs.csv

Keys:  Enter = finish sentence and save to notes.txt   c = clear   q = quit

Glosses are decided by detector.py: a gloss prints as soon as the model is
confident for a couple of predictions, or when a brief sign passes with a
clear peak. Holding a sign prints it once; the next sign can follow
immediately. When two glosses look alike and the model cannot separate them,
nothing is printed (tune the thresholds at the top of detector.py).
"""
import argparse
import csv
import threading
import time
from collections import deque

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from detector import GlossDetector
from features import N_MODEL_FEATURES, build_features
from utils import (ACTIONS_PATH, IDLE, MIN_WINDOW_FRAMES, MODEL_PATH, MU_PATH,
                   SEQ_LEN, SIGMA_PATH, WINDOW_SECONDS, draw_landmarks,
                   extract_keypoints, make_holistic, mediapipe_detection,
                   open_camera, resample_window)

MAX_GLOSSES = 8
SAVE_FILE = "notes.txt"

parser = argparse.ArgumentParser()
parser.add_argument("--grammar", action="store_true",
                    help="load the T5 model and convert glosses to a sentence")
parser.add_argument("--debug", action="store_true",
                    help="write every prediction to debug_probs.csv")
args = parser.parse_args()

# ------------------------------------------------------------------ load
# No try/except: if something is missing we want to stop, not silently continue.
model = load_model(MODEL_PATH)
mu = np.load(MU_PATH)
sigma = np.load(SIGMA_PATH)
actions = np.load(ACTIONS_PATH)
print("Classes:", list(actions))
if mu.shape[0] != N_MODEL_FEATURES:
    raise SystemExit(f"The saved model expects {mu.shape[0]} features but features.py "
                     f"produces {N_MODEL_FEATURES}. Run: python train.py")

tokenizer = grammar_model = None
if args.grammar:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    name = "vennify/t5-base-grammar-correction"
    tokenizer = AutoTokenizer.from_pretrained(name)
    grammar_model = AutoModelForSeq2SeqLM.from_pretrained(name)
    print("Grammar model loaded")


def glosses_to_sentence(glosses):
    """Stage 2 hook: gloss list -> English sentence."""
    if not glosses:
        return ""
    raw = " ".join(glosses)
    if grammar_model is None:
        return raw.capitalize() + "."
    inputs = tokenizer(f"grammar: {raw}", return_tensors="pt")
    out = grammar_model.generate(**inputs, max_length=128)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ------------------------------------------------------------------ state
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (200, 50, 200),
          (50, 200, 200), (200, 200, 50), (200, 50, 50)]
buffer = deque()                      # (timestamp, keypoints) for the last WINDOW_SECONDS
probs = np.zeros(len(actions))
glosses = []
detector = GlossDetector(actions, idle=IDLE)
state = {"sentence": "", "busy": False, "saved_until": 0.0}

debug_file = debug_writer = None
if args.debug:
    debug_file = open("debug_probs.csv", "w", newline="", encoding="utf-8")
    debug_writer = csv.writer(debug_file)
    debug_writer.writerow(["time"] + [str(a) for a in actions] + ["printed"])


def finish(gloss_list):
    text = glosses_to_sentence(gloss_list)
    with open(SAVE_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Saved: {text}")
    state["sentence"] = text
    state["saved_until"] = time.time() + 1.0
    state["busy"] = False


def draw_probs(image, probs):
    for i, p in enumerate(probs):
        cv2.rectangle(image, (0, 60 + i * 40), (int(p * 200), 90 + i * 40),
                      colors[i % len(colors)], -1)
        cv2.putText(image, str(actions[i]), (0, 85 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


cap = open_camera()
try:
    with make_holistic() as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            h, w = image.shape[:2]

            # Keep the last WINDOW_SECONDS of frames (including frames with no
            # hands = zeros), then resample to SEQ_LEN exactly like record.py.
            now = time.perf_counter()
            buffer.append((now, extract_keypoints(results)))
            while buffer and now - buffer[0][0] > WINDOW_SECONDS:
                buffer.popleft()

            window_ready = (len(buffer) >= MIN_WINDOW_FRAMES and
                            buffer[-1][0] - buffer[0][0] >= 0.8 * WINDOW_SECONDS)
            if window_ready:
                stamps = [t for t, _ in buffer]
                kps = [k for _, k in buffer]
                window = resample_window(stamps, kps, SEQ_LEN)
                x = (build_features(window) - mu) / sigma
                probs = model(x[None].astype(np.float32), training=False).numpy()[0]

                printed = []
                for label, p in detector.update(probs, now):
                    glosses.append(label)
                    glosses[:] = glosses[-MAX_GLOSSES:]
                    state["sentence"] = ""
                    printed.append(label)
                    print(f"gloss: {label} ({p:.2f})")
                if debug_writer:
                    debug_writer.writerow([f"{now:.3f}"] + [f"{v:.3f}" for v in probs]
                                          + [" ".join(printed)])

            draw_probs(image, probs)
            top = int(np.argmax(probs))
            cv2.putText(image, f"top: {actions[top]} {probs[top]:.2f}", (0, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (0, 0), (w, 40), (245, 117, 16), -1)
            cv2.putText(image, "Glosses: " + " ".join(glosses), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (0, h - 40), (w, h), (35, 35, 35), -1)
            text = "Translating..." if state["busy"] else state["sentence"]
            cv2.putText(image, text, (3, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)
            if time.time() < state["saved_until"]:
                cv2.putText(image, "SAVED!", (w // 2 - 80, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

            cv2.imshow("Sign translator (Enter=save, c=clear, q=quit)", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                glosses.clear()
                state["sentence"] = ""
            elif key == 13 and glosses and not state["busy"]:
                state["busy"] = True
                # Run off the video thread so the feed does not freeze.
                threading.Thread(target=finish, args=(list(glosses),), daemon=True).start()
                glosses.clear()
finally:
    cap.release()
    cv2.destroyAllWindows()
    if debug_file:
        debug_file.close()