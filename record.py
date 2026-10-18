"""Record training samples for one label.

Usage:
    python record.py hello          # record 30 samples of the sign
    python record.py idle           # record 30 samples of NOT signing
    python record.py hello --n 20   # record 20 more (appends, never overwrites)

Each sample is one file: Dataset/MP_Data_v2/<label>/<index>.npy with shape (30, 258).
Each sample covers WINDOW_SECONDS of real time (set in utils.py) and is
resampled to SEQ_LEN frames. A sample is only written once its window is
complete, so pressing 'q' mid-sample never leaves a half-saved sequence behind.

Tips for good data:
  * Record in several sessions (different lighting, distance, clothing).
  * 'idle' must be varied: standing still, hands down, hands raised but not
    signing, scratching your face, reaching for something, no hands visible.
    This is what lets the model say "no sign" in real time.
"""
import argparse
import os
import time

import cv2
import numpy as np

from utils import (ACTIONS, DATA_PATH, IDLE, MIN_WINDOW_FRAMES, SEQ_LEN,
                   WINDOW_SECONDS, draw_landmarks, extract_keypoints,
                   make_holistic, mediapipe_detection, open_camera,
                   resample_window)

COUNTDOWN_SECONDS = 2


def next_index(label_dir):
    existing = [int(f[:-4]) for f in os.listdir(label_dir)
                if f.endswith(".npy") and f[:-4].isdigit()]
    return max(existing) + 1 if existing else 0


def banner(image, text, color):
    cv2.rectangle(image, (0, 0), (image.shape[1], 50), color, -1)
    cv2.putText(image, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=ACTIONS)
    parser.add_argument("--n", type=int, default=30, help="new samples to record")
    args = parser.parse_args()

    label_dir = os.path.join(DATA_PATH, args.label)
    os.makedirs(label_dir, exist_ok=True)
    start = next_index(label_dir)
    hint = "do NOT sign (vary what you do)" if args.label == IDLE else "perform the sign"
    print(f"Recording '{args.label}' from sample #{start}. Press 'q' to stop.")

    cap = open_camera()
    try:
        with make_holistic() as holistic:
            for i in range(start, start + args.n):
                # ---- get ready
                t0 = time.time()
                while time.time() - t0 < COUNTDOWN_SECONDS:
                    ret, frame = cap.read()
                    if not ret:
                        return
                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)
                    left = int(COUNTDOWN_SECONDS - (time.time() - t0)) + 1
                    banner(image, f"{args.label.upper()} #{i}  -  {hint}", (50, 50, 50))
                    cv2.putText(image, f"GET READY: {left}", (160, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4, cv2.LINE_AA)
                    cv2.imshow("Record", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return

                # ---- record WINDOW_SECONDS of real time, then resample to SEQ_LEN
                stamps, frames = [], []
                t_start = time.perf_counter()
                while time.perf_counter() - t_start < WINDOW_SECONDS:
                    ret, frame = cap.read()
                    if not ret:
                        return
                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)
                    stamps.append(time.perf_counter())
                    frames.append(extract_keypoints(results))
                    elapsed = time.perf_counter() - t_start
                    banner(image, f"{args.label.upper()} #{i}  -  {hint}", (0, 0, 180))
                    cv2.putText(image, f"RECORDING {elapsed:.2f}/{WINDOW_SECONDS}s", (60, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.imshow("Record", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Stopped; the unfinished sample was discarded.")
                        return

                if len(frames) < MIN_WINDOW_FRAMES:
                    print(f"Only {len(frames)} real frames in {WINDOW_SECONDS}s - too few, "
                          f"sample discarded. Close other apps / lower camera resolution.")
                    continue

                sample = resample_window(stamps, frames, SEQ_LEN)
                np.save(os.path.join(label_dir, f"{i}.npy"), sample)
                fps = (len(frames) - 1) / max(stamps[-1] - stamps[0], 1e-6)
                print(f"saved {args.label}/{i}.npy  ({len(frames)} real frames, ~{fps:.0f} fps -> {SEQ_LEN})")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()