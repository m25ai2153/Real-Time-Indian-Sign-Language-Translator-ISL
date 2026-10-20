"""Real-time sign recognition from the webcam.

Usage:
    python realtime.py              # glosses only
    python realtime.py --grammar    # also turn the gloss sequence into an English sentence
    python realtime.py --debug      # also log every prediction to debug_probs.csv
    python realtime.py --save-run run1.npz   # also keep the raw frames, to replay and tune offline
    python realtime.py --no-gate    # compare without the rest gate (see gate.py)

Keys:  Enter = finish sentence and save to notes.txt   c = clear   q = quit

Glosses are decided by detector.py: a gloss prints as soon as the model is
confident for a couple of predictions, or when a brief sign passes with a
clear peak. Holding a sign prints it once; the next sign can follow
immediately. When two glosses look alike and the model cannot separate them,
nothing is printed (tune the thresholds at the top of detector.py).

Speed: the camera loop (MediaPipe) and the model run in separate threads.
Before, every frame waited for the model, so the loop ran at ~5 fps while the
training data was recorded at ~18 fps; a sign then reached the model as just a
few frames. Now the loop only captures and draws, so frames arrive at the same
rate as when recording, and the model predicts on the newest window as fast as
it can. The top-left of the picture shows both rates.
"""
import argparse
import csv
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from detector import GlossDetector
from features import N_MODEL_FEATURES, build_features
from gate import rest_gate
from stream import GlossStream
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
parser.add_argument("--save-run", metavar="FILE", default=None,
                    help="save every captured frame (timestamps + keypoints) to this .npz file")
parser.add_argument("--no-gate", action="store_true",
                    help="switch off the rest gate (no hand in the window / no arm movement = idle)")
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


@tf.function(reduce_retracing=True)
def predict_fn(x):
    return model(x, training=False)


predict_fn(tf.zeros((1, SEQ_LEN, N_MODEL_FEATURES), tf.float32))   # build once, not during the first sign

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
lock = threading.Lock()               # guards buffer, probs, stream and the counters below
new_frame = threading.Event()
stop = threading.Event()
probs = np.zeros(len(actions))
idle_probs = np.zeros(len(actions))
idle_probs[list(map(str, actions)).index(IDLE)] = 1.0
stream = GlossStream(max_items=MAX_GLOSSES)       # what goes into the sentence (see stream.py)
detector = GlossDetector(actions, idle=IDLE)          # only the inference thread touches it
state = {"sentence": "", "busy": False, "saved_until": 0.0,
         "frames": 0, "preds": 0, "error": None}

debug_file = debug_writer = None
if args.debug:
    debug_file = open("debug_probs.csv", "w", newline="", encoding="utf-8")
    debug_writer = csv.writer(debug_file)
    debug_writer.writerow(["time"] + [str(a) for a in actions] + ["gate", "printed"])

run_stamps, run_kps = [], []          # only filled with --save-run


def finish(gloss_list):
    text = glosses_to_sentence(gloss_list)
    with open(SAVE_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Saved: {text}")
    state["sentence"] = text
    state["saved_until"] = time.time() + 1.0
    state["busy"] = False


def inference_loop():
    """Predict on the newest window as often as the model allows, independent of the camera."""
    global probs
    last_stamp = None
    try:
        while not stop.is_set():
            if not new_frame.wait(0.1):
                continue
            new_frame.clear()
            with lock:                                  # snapshot, so the camera keeps going
                if (not buffer or buffer[-1][0] == last_stamp or len(buffer) < MIN_WINDOW_FRAMES
                        or buffer[-1][0] - buffer[0][0] < 0.8 * WINDOW_SECONDS):
                    continue
                snap = list(buffer)
            last_stamp = snap[-1][0]
            stamps = [t for t, _ in snap]
            window = resample_window(stamps, [k for _, k in snap], SEQ_LEN)
            x = (build_features(window) - mu) / sigma
            raw = predict_fn(tf.constant(x[None], tf.float32)).numpy()[0]
            gated = (not args.no_gate) and rest_gate(window)[0]
            p = idle_probs if gated else raw           # the detector and the display see the gated result

            events = detector.update(p, last_stamp)
            printed, lines = [], []
            with lock:
                probs = p
                for label, prob in events:
                    printed.append(label)
                    kept, note = stream.push(label, last_stamp)
                    lines.append(f"gloss: {label} ({prob:.2f})" if kept else f"  skip: {label} ({note})")
                    if kept:
                        state["sentence"] = ""
                state["preds"] += 1
            for line in lines:
                print(line)
            if debug_writer:
                debug_writer.writerow([f"{last_stamp:.3f}"] + [f"{v:.3f}" for v in raw]   # raw model output
                                      + [int(gated), " ".join(printed)])
    except Exception:                                   # show the real error, then stop everything
        state["error"] = traceback.format_exc()
        stop.set()


def draw_probs(image, p):
    for i, v in enumerate(p):
        cv2.rectangle(image, (0, 60 + i * 40), (int(v * 200), 90 + i * 40),
                      colors[i % len(colors)], -1)
        cv2.putText(image, str(actions[i]), (0, 85 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


worker = threading.Thread(target=inference_loop, daemon=True)
worker.start()
t_start = time.perf_counter()
last_rate = (t_start, 0, 0)
rate_text = ""

cap = open_camera()
try:
    with make_holistic() as holistic:
        while cap.isOpened() and not stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            h, w = image.shape[:2]

            # Keep the last WINDOW_SECONDS of frames (including frames with no
            # hands = zeros); the inference thread resamples them to SEQ_LEN
            # exactly like record.py does.
            now = time.perf_counter()
            kp = extract_keypoints(results)
            with lock:
                buffer.append((now, kp))
                while buffer and now - buffer[0][0] > WINDOW_SECONDS:
                    buffer.popleft()
                state["frames"] += 1
                cur_probs = probs.copy()
                cur_glosses = stream.labels
            if args.save_run:
                run_stamps.append(now)
                run_kps.append(kp)
            new_frame.set()

            if now - last_rate[0] >= 1.0:               # camera and model rates, refreshed every second
                dt = now - last_rate[0]
                rate_text = (f"cam {(state['frames'] - last_rate[1]) / dt:.0f} fps | "
                             f"model {(state['preds'] - last_rate[2]) / dt:.0f}/s")
                last_rate = (now, state["frames"], state["preds"])

            draw_probs(image, cur_probs)
            top = int(np.argmax(cur_probs))
            cv2.putText(image, f"top: {actions[top]} {cur_probs[top]:.2f}", (0, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (0, 0), (w, 40), (245, 117, 16), -1)
            cv2.putText(image, "Glosses: " + " ".join(cur_glosses), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, rate_text, (w - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
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
                with lock:
                    stream.clear()
                state["sentence"] = ""
            elif key == 13 and cur_glosses and not state["busy"]:
                state["busy"] = True
                # Run off the video thread so the feed does not freeze.
                threading.Thread(target=finish, args=(list(cur_glosses),), daemon=True).start()
                with lock:
                    stream.clear()
finally:
    stop.set()
    worker.join(timeout=2.0)
    cap.release()
    cv2.destroyAllWindows()
    if debug_file:
        debug_file.close()
    secs = max(time.perf_counter() - t_start, 1e-6)
    print(f"Average: camera {state['frames'] / secs:.1f} fps, model {state['preds'] / secs:.1f} predictions/s")
    if args.save_run and run_stamps:
        np.savez_compressed(args.save_run, stamps=np.asarray(run_stamps) - run_stamps[0],
                            keypoints=np.asarray(run_kps, dtype=np.float32))
        print(f"Saved {len(run_stamps)} frames to {args.save_run}")
if state["error"]:
    raise SystemExit("The recognition thread crashed:\n" + state["error"])