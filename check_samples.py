"""Find recorded samples that cannot contain their sign (numpy only).

Usage:
    python check_samples.py                  # report only
    python check_samples.py --quarantine     # also move the suspicious files to Dataset/_rejected/<gloss>/
    python check_samples.py path             # default path: Dataset/MP_Data_v2

A sample of a sign is suspicious when the rest gate (gate.py) calls it rest: no hand
tracked in (almost) the whole window, or both wrists hardly moving. Typical causes:
  * the cue was much longer than the sign (record_sequence centres the window on the
    middle of the cue, so a 0.5 s sign inside a 2 s cue leaves the window showing rest),
  * the hand left the camera picture,
  * you waited for the cue and signed late or early.
Nothing is deleted: --quarantine only moves files, and idle is never checked.
"""
import argparse
import os
import shutil

import numpy as np

from features import STORED_FEATURES
from gate import MIN_HAND_FRACTION, MIN_WRIST_MOTION, rest_gate

parser = argparse.ArgumentParser()
parser.add_argument("path", nargs="?", default=os.path.join("Dataset", "MP_Data_v2"))
parser.add_argument("--quarantine", action="store_true",
                    help="move the suspicious samples to <path>/../_rejected/<gloss>/")
parser.add_argument("--idle", default="idle", help="name of the rest class (never checked)")
args = parser.parse_args()

reject_root = os.path.join(os.path.dirname(os.path.normpath(args.path)), "_rejected")
print(f"rest gate: hand tracked < {MIN_HAND_FRACTION:.0%} of the window or wrist movement < {MIN_WRIST_MOTION} shoulder widths")
print(f"{'class':>10} {'samples':>8} {'suspect':>8} {'share':>6} {'median motion':>14} {'median hand':>12}")
total = flagged_total = 0
for name in sorted(os.listdir(args.path)):
    folder = os.path.join(args.path, name)
    if not os.path.isdir(folder) or name == args.idle:
        continue
    rows = []
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".npy"):
            continue
        s = np.load(os.path.join(folder, f))
        if s.ndim == 2 and s.shape[1] == STORED_FEATURES:
            rows.append((f, *rest_gate(s)))
    if not rows:
        continue
    bad = [r for r in rows if r[1]]
    total += len(rows)
    flagged_total += len(bad)
    print(f"{name:>10} {len(rows):>8} {len(bad):>8} {len(bad) / len(rows):>6.0%} "
          f"{np.median([r[3] for r in rows]):>14.2f} {np.median([r[2] for r in rows]):>12.0%}")
    for f, _, hand, motion in bad[:8]:
        print(f"{'':>10}   {f}: hand {hand:.0%}, motion {motion:.2f}")
    if len(bad) > 8:
        print(f"{'':>10}   ... and {len(bad) - 8} more")
    if args.quarantine and bad:
        dest = os.path.join(reject_root, name)
        os.makedirs(dest, exist_ok=True)
        for f, *_ in bad:
            shutil.move(os.path.join(folder, f), os.path.join(dest, f))
print(f"\n{flagged_total} of {total} sign samples look like rest."
      + (f" Moved to {reject_root}." if args.quarantine and flagged_total else
         " Run again with --quarantine to move them out of the training data." if flagged_total else ""))
