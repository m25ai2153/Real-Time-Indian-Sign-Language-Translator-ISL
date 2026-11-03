"""Real-time sign translator: webcam -> signs (glosses) -> English sentence.

Usage:
    python realtime.py                       # the demo screen
    python realtime.py --save-run run6.npz   # also keep the raw frames, to replay and tune offline
    python realtime.py --debug               # also log every prediction to debug_probs.csv
    python realtime.py --no-gate             # compare without the rest gate (see gate.py)
    python realtime.py --pause 5             # seconds of silence that end a sentence (default 4)
    python realtime.py --no-translate        # show plain glosses instead of the English sentence

Keys:  Enter = finish the sentence now (saved to notes.txt)   c = clear   p = show/hide probabilities   q = quit

Pipeline: MediaPipe landmarks -> sign model (train.py) -> glosses (detector.py, stream.py)
-> English sentence (translator.py, a small T5 fine-tuned by train_gloss_t5.py).
Screen (ui.py): video top-left, glosses top-right, translation along the bottom.

Threads: the camera loop only captures and draws (so frames arrive at the rate used when
recording); one thread runs the sign model on the newest 1-second window; one thread runs
the translator whenever the glosses change, so a slow translation never freezes the video.
A pause of --pause seconds after the last sign ends the sentence: it is saved to notes.txt
and moves to the small history under the translation, and the next sign starts a new one.
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

import ui
from detector import GlossDetector
from features import N_MODEL_FEATURES, build_features
from gate import rest_gate
from stream import GlossStream
from translator import DEFAULT_DIR, GlossTranslator, plain
from utils import (ACTIONS_PATH, IDLE, MIN_WINDOW_FRAMES, MODEL_PATH, MU_PATH,
                   SEQ_LEN, SIGMA_PATH, WINDOW_SECONDS, draw_landmarks,
                   extract_keypoints, make_holistic, mediapipe_detection,
                   open_camera, resample_window)

MAX_GLOSSES = 14                      # glosses kept in one sentence
SAVE_FILE = "notes.txt"

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="write every prediction to debug_probs.csv")
parser.add_argument("--save-run", metavar="FILE", default=None,
                    help="save every captured frame (timestamps + keypoints) to this .npz file")
parser.add_argument("--no-gate", action="store_true",
                    help="switch off the rest gate (no hand in the window = idle)")
parser.add_argument("--pause", type=float, default=4.0,
                    help="seconds without a sign that end the sentence (default 4)")
parser.add_argument("--translator-dir", default=DEFAULT_DIR, help="folder written by train_gloss_t5.py")
parser.add_argument("--translator-python", default=None,
                    help="python.exe of another environment that has torch + transformers (default: this one)")
parser.add_argument("--no-translate", action="store_true", help="show plain glosses, no translator")
args = parser.parse_args()

# ------------------------------------------------------------------ load
# No try/except for the sign model: if something is missing we want to stop, not silently continue.
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

translator = GlossTranslator(args.translator_dir if not args.no_translate else "<none>", python=args.translator_python)
print("Translator:", "fine-tuned T5 loaded" if translator.ready else translator.note or "plain glosses")

# ------------------------------------------------------------------ state
buffer = deque()                      # (timestamp, keypoints) for the last WINDOW_SECONDS
lock = threading.Lock()               # guards buffer, probs, stream, trans, history and the counters
new_frame = threading.Event()
gloss_changed = threading.Event()
stop = threading.Event()
probs = np.zeros(len(actions))
idle_probs = np.zeros(len(actions))
idle_probs[list(map(str, actions)).index(IDLE)] = 1.0
stream = GlossStream(max_items=MAX_GLOSSES)           # what goes into the sentence (see stream.py)
detector = GlossDetector(actions, idle=IDLE)          # only the inference thread touches it
trans = {"glosses": (), "text": "", "pending": False}  # the translation currently on screen
history = []                                           # finished sentences (English)
state = {"frames": 0, "preds": 0, "last_event": None, "saved_until": 0.0, "error": None}

debug_file = debug_writer = None
if args.debug:
    debug_file = open("debug_probs.csv", "w", newline="", encoding="utf-8")
    debug_writer = csv.writer(debug_file)
    debug_writer.writerow(["time"] + [str(a) for a in actions] + ["gate", "printed"])

run_stamps, run_kps = [], []          # only filled with --save-run


def finalize(labels):
    """Translate a finished sentence (the translator cache makes this instant), save it, add it to the history."""
    if not labels:
        return
    try:
        text = translator.translate(labels, announce=True)
    except Exception:
        text = plain(labels)
    with open(SAVE_FILE, "a", encoding="utf-8") as f:
        f.write(f"{' '.join(labels)} | {text}\n")
    print(f"Sentence: {' '.join(labels)}  ->  {text}")
    with lock:
        history.append(text)
        state["saved_until"] = time.time() + 1.0


def end_sentence(save=True):
    """Clear the current sentence (and save it unless save=False). Safe to call from any thread."""
    with lock:
        labels = stream.labels
        stream.clear()
        trans.update(glosses=(), text="", pending=False)
    if save and labels:
        threading.Thread(target=finalize, args=(labels,), daemon=True).start()


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
            printed, lines, finished = [], [], []
            with lock:
                probs = p
                # a long silence ends the previous sentence before a new sign starts the next one
                if (events and stream.labels and state["last_event"] is not None
                        and last_stamp - state["last_event"] > args.pause):
                    finished = stream.labels
                    stream.clear()
                    trans.update(glosses=(), text="", pending=False)
                for label, prob in events:
                    printed.append(label)
                    state["last_event"] = last_stamp
                    kept, note = stream.push(label, last_stamp)
                    lines.append(f"gloss: {label} ({prob:.2f})" if kept else f"  skip: {label} ({note})")
                if events:
                    gloss_changed.set()
                state["preds"] += 1
            if finished:
                threading.Thread(target=finalize, args=(finished,), daemon=True).start()
            for line in lines:
                print(line)
            if debug_writer:
                debug_writer.writerow([f"{last_stamp:.3f}"] + [f"{v:.3f}" for v in raw]   # raw model output
                                      + [int(gated), " ".join(printed)])
    except Exception:                                   # show the real error, then stop everything
        state["error"] = traceback.format_exc()
        stop.set()


def translate_loop():
    """Translate the current glosses whenever they change; the camera loop never waits for this."""
    while not stop.is_set():
        if not gloss_changed.wait(0.2):
            continue
        gloss_changed.clear()
        with lock:
            want = tuple(stream.labels)
            if want == trans["glosses"]:
                continue
            trans["pending"] = bool(want)
        try:
            text = translator.translate(list(want)) if want else ""
        except Exception:
            text = plain(want)
        with lock:
            if tuple(stream.labels) == want:            # still current
                trans.update(glosses=want, text=text, pending=False)
            else:                                       # glosses moved on while translating
                gloss_changed.set()


workers = [threading.Thread(target=inference_loop, daemon=True),
           threading.Thread(target=translate_loop, daemon=True)]
for t in workers:
    t.start()
t_start = time.perf_counter()
last_rate = (t_start, 0, 0)
rate_text = ""
show_probs = False

cap = open_camera()
try:
    with make_holistic() as holistic:
        while cap.isOpened() and not stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            # Keep the last WINDOW_SECONDS of frames (including frames with no hands = zeros);
            # the inference thread resamples them to SEQ_LEN exactly like record.py does.
            now = time.perf_counter()
            kp = extract_keypoints(results)
            with lock:
                buffer.append((now, kp))
                while buffer and now - buffer[0][0] > WINDOW_SECONDS:
                    buffer.popleft()
                state["frames"] += 1
                cur_probs = probs.copy()
                cur_glosses = stream.labels
                cur_text, cur_pending = trans["text"], trans["pending"]
                cur_history = list(history)
                saved = time.time() < state["saved_until"]
            if args.save_run:
                run_stamps.append(now)
                run_kps.append(kp)
            new_frame.set()

            if now - last_rate[0] >= 1.0:               # camera and model rates, refreshed every second
                dt = now - last_rate[0]
                rate_text = (f"cam {(state['frames'] - last_rate[1]) / dt:.0f} fps | "
                             f"model {(state['preds'] - last_rate[2]) / dt:.0f}/s")
                last_rate = (now, state["frames"], state["preds"])

            screen = ui.compose(image, cur_glosses, cur_text, pending=cur_pending, history=cur_history,
                                info=rate_text, probs=cur_probs if show_probs else None,
                                actions=actions, saved=saved)
            cv2.imshow("Sign translator  (Enter = finish sentence, c = clear, p = probabilities, q = quit)", screen)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                end_sentence(save=False)
            elif key == ord("p"):
                show_probs = not show_probs
            elif key == 13:
                end_sentence(save=True)
finally:
    stop.set()
    for t in workers:
        t.join(timeout=2.0)
    if stream.labels:                                   # keep the unfinished sentence
        finalize(stream.labels)
    translator.close()
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
