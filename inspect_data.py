"""Quick data health check (numpy only).

Usage:
    python inspect_data.py [path]          # default: Dataset/MP_Data_v2

For every class it prints the number of samples, how often each hand was
actually tracked, and the average finger-extension ratios (thumb, index,
middle, ring, pinky; ~1.0+ = finger straight, small = curled).

'solo' = samples from record_sequence.py with a single sign (a clean example with rest
around it); 'seq' = samples from record_sequence.py with signs back to back. If a sign
you will use in conversation has 0 under 'seq', the model has never seen it next to
another sign.
'arm up' = share of frames where a wrist is above the shoulder line, 'motion' =
how far the higher wrist moves inside one window (in shoulder widths). A true
rest ('idle') sample should have arm up near 0% and motion near 0; signs should
show a clearly raised arm and clear motion.

Use it to see WHY two signs get confused:
  * if 'hello' and 'head' have similar ratios, the recordings do not show the
    hand shape clearly (hand too far, blurred, or tracking lost);
  * if a hand is tracked in only 50% of frames, the model has little to learn from;
  * if 'idle' has far fewer samples than the signs, it will rarely be predicted.
"""
import argparse
import os

import numpy as np

from features import STORED_FEATURES, build_features

parser = argparse.ArgumentParser()
parser.add_argument("path", nargs="?", default=os.path.join("Dataset", "MP_Data_v2"))
args = parser.parse_args()

np.set_printoptions(precision=2, suppress=True)
print(f"{'class':>10} {'samples':>8} {'solo':>5} {'seq':>5} {'L hand':>7} {'R hand':>7} {'no pose':>8} {'arm up':>7} {'motion':>7}   mean extension (thumb index middle ring pinky)")
for name in sorted(os.listdir(args.path)):
    folder = os.path.join(args.path, name)
    if not os.path.isdir(folder):
        continue
    samples = [np.load(os.path.join(folder, f)) for f in sorted(os.listdir(folder))
               if f.endswith(".npy")]
    samples = [s for s in samples if s.ndim == 2 and s.shape[1] == STORED_FEATURES]
    if not samples:
        print(f"{name:>10} {'0':>8}")
        continue
    X = np.stack(samples)                                   # (N, T, 258)
    left = np.any(X[..., 132:195] != 0, axis=-1).mean()
    right = np.any(X[..., 195:258] != 0, axis=-1).mean()
    no_pose = (~np.any(X[..., :132] != 0, axis=-1)).mean()
    # Arm height from the pose wrists (available even when the hand is not tracked).
    # y is in shoulder widths below the shoulder line, so negative = above the shoulders.
    wrist_y = np.stack([X[..., 4 * 15 + 1], X[..., 4 * 16 + 1]], axis=-1).min(axis=-1)  # higher wrist, (N, T)
    arm_up = (wrist_y < 0).mean()
    motion = (wrist_y.max(axis=1) - wrist_y.min(axis=1)).mean()
    feats = build_features(X, fill_gaps=False)              # (N, T, features), raw tracking only
    ext = feats[..., -10:].reshape(-1, 10)                  # left 5 | right 5
    rows = []
    for hand in (ext[:, :5], ext[:, 5:]):
        present = hand.sum(axis=1) > 0
        if present.any():
            rows.append((present.sum(), hand[present].mean(axis=0)))
    mean_ext = (max(rows, key=lambda r: r[0])[1] if rows else np.zeros(5))
    n_seq = sum(f.startswith("seq_") for f in os.listdir(folder))
    n_solo = sum(f.startswith("solo_") for f in os.listdir(folder))
    print(f"{name:>10} {len(samples):>8} {n_solo:>5} {n_seq:>5} {left:>7.0%} {right:>7.0%} {no_pose:>8.0%} "
          f"{arm_up:>7.0%} {motion:>7.2f}   {mean_ext}")