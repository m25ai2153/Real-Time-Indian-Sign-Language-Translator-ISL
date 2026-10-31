"""List recorded samples whose label the trained model strongly disagrees with.

Usage (after python train.py, from the project folder):
    python review_labels.py                    # report + label_review.csv
    python review_labels.py --conf 0.7         # looser: also show less certain disagreements
    python review_labels.py --move             # also move the suspects to Dataset/_review/<label>/

A sample is a suspect when the model puts at least --conf (default 0.8) on a DIFFERENT sign.
Nothing is deleted: --move only moves files into Dataset/_review, and you can move them back.

Read the result with care:
  * A suspect may be a genuinely wrong sign (you slipped, or signed the next sign early),
    or a sequence window dominated by the neighbouring sign: record_sequence cuts one window
    per cue, centred on the cue, so a short sign next to a long one can look like the long one.
  * Two signs that really look alike will also show up here. Many suspects between the same
    pair (e.g. I -> doctor) points at look-alike signs, not at bad recordings.
  * Look before you move: python view_sample.py --csv label_review.csv plays each suspect.
The model was trained on these samples, so it can only flag the clearest errors.
"""
import argparse
import csv
import os
import shutil
from collections import Counter

import numpy as np
from tensorflow.keras.models import load_model

from features import build_features
from utils import (ACTIONS_PATH, DATA_PATH, MODEL_PATH, MU_PATH, N_FEATURES, SEQ_LEN,
                   SIGMA_PATH)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=DATA_PATH)
ap.add_argument("--conf", type=float, default=0.8, help="probability of the wrong sign needed to flag a sample")
ap.add_argument("--top", type=int, default=30, help="how many suspects to print")
ap.add_argument("--move", action="store_true", help="move suspects to <data>/../_review/<label>/")
ap.add_argument("--csv", default="label_review.csv")
args = ap.parse_args()

model = load_model(MODEL_PATH)
mu, sigma = np.load(MU_PATH), np.load(SIGMA_PATH)
actions = [str(a) for a in np.load(ACTIONS_PATH)]

names, labels, feats = [], [], []
for idx, action in enumerate(actions):
    folder = os.path.join(args.data, action)
    for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if not f.endswith(".npy"):
            continue
        s = np.load(os.path.join(folder, f))
        if s.shape != (SEQ_LEN, N_FEATURES):
            continue
        names.append((action, f)); labels.append(idx)
        feats.append((build_features(s) - mu) / sigma)
X = np.asarray(feats, np.float32)
P = np.concatenate([model.predict(X[i:i + 256], verbose=0) for i in range(0, len(X), 256)])
pred = P.argmax(1)
labels = np.asarray(labels)

rows = []
for (action, f), lab, p in zip(names, labels, P):
    top = int(p.argmax())
    if top != lab and p[top] >= args.conf:
        rows.append((action, f, actions[top], float(p[top]), float(p[lab])))
rows.sort(key=lambda r: -r[3])

print(f"{len(names)} samples checked, {len(rows)} flagged (model puts >= {args.conf:.0%} on another sign)\n")
per = Counter(r[0] for r in rows); tot = Counter(a for a, _ in names)
print(f"{'sign':>10} {'samples':>8} {'flagged':>8}")
for a in actions:
    print(f"{a:>10} {tot[a]:>8} {per[a]:>8}")
print("\nMost common mix-ups (labelled as -> model says):")
for (a, b), n in Counter((r[0], r[2]) for r in rows).most_common(8):
    print(f"   {a} -> {b}: {n}")
print(f"\nStrongest {min(args.top, len(rows))} suspects:")
for a, f, b, p, own in rows[:args.top]:
    print(f"   {a}/{f}: model says {b} ({p:.0%}), {a} only {own:.0%}")

with open(args.csv, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["label", "file", "model_says", "p_model", "p_label"])
    for r in rows:
        w.writerow([r[0], r[1], r[2], f"{r[3]:.3f}", f"{r[4]:.3f}"])
print(f"\nSaved {args.csv}. Next: python view_sample.py --csv {args.csv}")

if args.move and rows:
    root = os.path.join(os.path.dirname(os.path.normpath(args.data)), "_review")
    for a, f, *_ in rows:
        os.makedirs(os.path.join(root, a), exist_ok=True)
        shutil.move(os.path.join(args.data, a, f), os.path.join(root, a, f))
    print(f"Moved {len(rows)} samples to {root}. Retrain with: python train.py")
