"""Build the resting-left-hand pool used by train.py (numpy only).

Usage:
    python make_left_pool.py run3.npz [more_runs.npz ...] [--out Dataset/left_pool.npz]

Use recordings made with  python realtime.py --save-run NAME.npz  in which your LEFT hand
rested in view (on the desk or keyboard) while you signed with the right hand, the way it
happens when you sit at a laptop. The pool is a set of real left-hand + left-arm frames.
train.py adds them to right-handed training samples, and removes them from some idle
samples, so a visible left hand stops meaning "idle" to the model.
"""
import argparse
import os

import numpy as np

from features import LeftPool

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+", help=".npz files written by realtime.py --save-run")
parser.add_argument("--out", default=os.path.join("Dataset", "left_pool.npz"))
parser.add_argument("--min-len", type=int, default=32,
                    help="shortest stretch (frames) with the left hand tracked all the time")
args = parser.parse_args()

kps = []
for path in args.runs:
    kp = np.load(path)["keypoints"]
    share = float(np.any(kp[:, 132:195] != 0, axis=1).mean())
    print(f"{path}: {len(kp)} frames, left hand tracked in {share:.0%}")
    kps.append(kp)
pool = LeftPool.from_runs(kps, min_len=args.min_len)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
pool.save(args.out)
print(f"Saved {len(pool.frames)} frames in {len(pool.lengths)} stretches to {args.out}")
