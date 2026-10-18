"""Real-time sign recognition from the webcam.

Usage:
    python realtime.py              # glosses only (fine for one word)
    python realtime.py --grammar    # also turn the gloss sequence into an English sentence

Keys:  Enter = finish sentence and save to notes.txt   c = clear   q = quit

A gloss is added when the model is confident and stable for several frames.
To repeat the same sign, return to a neutral pose first (the 'idle' class
re-arms the detector).
"""
import argparse
import threading
import time
from collections import deque

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from utils import (ACTIONS_PATH, IDLE, MIN_WINDOW_FRAMES, MODEL_PATH, MU_PATH,
                   SEQ_LEN, SIGMA_PATH, WINDOW_SECONDS, draw_landmarks,
                   extract_keypoints, make_holistic, mediapipe_detection,
                   open_camera, resample_window)

THRESHOLD = 0.85      # minimum confidence
STABLE_FRAMES = 6     # consecutive predictions that must agree
MAX_GLOSSES = 8
SAVE_FILE = "notes.txt"

parser = argparse.ArgumentParser()
parser.add_argument("--grammar", action="store_true",
                    help="load the T5 model and convert glosses to a sentence")
args = parser.parse_args()

# ------------------------------------------------------------------ load
# No try/except: if something is missing we want to stop, not silently continue.
model = load_model(MODEL_PATH)
mu = np.load(MU_PATH)
sigma = np.load(SIGMA_PATH)
actions = np.load(ACTIONS_PATH)
print("Classes:", list(actions))

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
recent = deque(maxlen=STABLE_FRAMES)
probs = np.zeros(len(actions))
glosses = []
last_committed = None
state = {"sentence": "", "busy": False, "saved_until": 0.0}


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
                x = (resample_window(stamps, kps, SEQ_LEN) - mu) / sigma
                probs = model(x[None].astype(np.float32), training=False).numpy()[0]
                idx = int(np.argmax(probs))
                recent.append(idx if probs[idx] >= THRESHOLD else -1)

                if len(recent) == STABLE_FRAMES and len(set(recent)) == 1 and recent[-1] != -1:
                    label = str(actions[recent[-1]])
                    if label == IDLE:
                        last_committed = None            # re-arm
                    elif label != last_committed:
                        glosses.append(label)
                        glosses[:] = glosses[-MAX_GLOSSES:]
                        last_committed = label
                        state["sentence"] = ""

            draw_probs(image, probs)

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
            key = cv2.waitKey(10) & 0xFF
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