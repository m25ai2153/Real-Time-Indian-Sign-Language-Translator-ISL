"""Play recorded samples as a stick figure, to judge by eye whether a sample is the sign it is filed under.

    python view_sample.py Dataset/MP_Data_v2/I/seq_12.npy
    python view_sample.py --csv label_review.csv        # step through the suspects from review_labels.py
    python view_sample.py --folder Dataset/MP_Data_v2/doctor   # step through a whole folder

Keys:  n / Right = next    p / Left = previous    space = pause    m = move this sample to Dataset/_rejected/<label>/    q = quit
A sample is one second resampled to 32 frames, so it is shown slowly and looped. Body and hands are drawn
relative to your shoulders, exactly as the model sees them. A sample with no hand drawn means no hand was tracked.
"""
import argparse
import csv
import os
import shutil

import cv2
import numpy as np

from utils import DATA_PATH

POSE_LINES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (0, 11), (0, 12)]
HAND_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
SIZE, SCALE, CX, CY = 640, 125.0, 320, 230


def px(x, y):
    return int(CX + x * SCALE), int(CY + y * SCALE)


def draw(frame, title):
    img = np.full((SIZE, SIZE, 3), 30, np.uint8)
    pose = frame[:132].reshape(33, 4)[:, :2]
    for a, b in POSE_LINES:
        cv2.line(img, px(*pose[a]), px(*pose[b]), (200, 200, 200), 3)
    for name, sl, colour in (("left", slice(132, 195), (80, 200, 255)), ("right", slice(195, 258), (120, 255, 120))):
        hand = frame[sl].reshape(21, 3)[:, :2]
        if np.abs(hand).sum() == 0:
            continue
        for a, b in HAND_LINES:
            cv2.line(img, px(*hand[a]), px(*hand[b]), colour, 2)
        for pt in hand:
            cv2.circle(img, px(*pt), 3, colour, -1)
    cv2.putText(img, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "left hand = orange, right hand = green (as in the camera picture, not mirrored)",
                (10, SIZE - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--csv"); ap.add_argument("--folder")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--fps", type=float, default=8.0)
    a = ap.parse_args()
    items = []                                              # (label, file, note)
    if a.csv:
        with open(a.csv, encoding="utf-8") as fh:
            items = [(r["label"], r["file"], f"model says {r['model_says']} ({float(r['p_model']):.0%})")
                     for r in csv.DictReader(fh)]
    elif a.folder:
        label = os.path.basename(os.path.normpath(a.folder))
        items = [(label, f, "") for f in sorted(os.listdir(a.folder)) if f.endswith(".npy")]
        a.data = os.path.dirname(os.path.normpath(a.folder))
    elif a.path:
        items = [(os.path.basename(os.path.dirname(a.path)), os.path.basename(a.path), "")]
        a.data = os.path.dirname(os.path.dirname(a.path))
    else:
        ap.error("give a .npy file, --csv or --folder")
    i, t, paused = 0, 0, False
    while 0 <= i < len(items):
        label, f, note = items[i]
        src = os.path.join(a.data, label, f)
        if not os.path.exists(src):
            i += 1; continue
        s = np.load(src)
        title = f"{i + 1}/{len(items)}  {label}/{f}  {note}"
        while True:
            cv2.imshow("sample", draw(s[t % len(s)], title))
            key = cv2.waitKey(int(1000 / a.fps)) & 0xFF
            if not paused:
                t += 1
            if key in (ord("n"), 83):
                i += 1; t = 0; break
            if key in (ord("p"), 81):
                i = max(i - 1, 0); t = 0; break
            if key == ord(" "):
                paused = not paused
            if key == ord("m"):
                dest = os.path.join(os.path.dirname(os.path.normpath(a.data)), "_rejected", label)
                os.makedirs(dest, exist_ok=True)
                shutil.move(src, os.path.join(dest, f))
                print(f"moved {label}/{f} to {dest}")
                i += 1; t = 0; break
            if key == ord("q"):
                return
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
