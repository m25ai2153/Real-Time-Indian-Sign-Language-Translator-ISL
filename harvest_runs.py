"""Turn the stray signs of live runs into "idle" training samples (hard negatives).

A live run saved with  python realtime.py --save-run runX.npz  contains every frame. This
script replays it through the current model, gate, detector and stream (exactly the live
chain), lines the glosses it found up against the sentences you signed, and picks out the
glosses that were NOT signed: the model's stray prints between two signs (a confident
"yes" after "no", an "I" after "thank_you doctor"). The windows where those strays fired
are saved as idle samples, so the next training run learns that this movement is not a sign.

    python harvest_runs.py run6_outdoor.npz run6_indoor.npz run7_indoor.npz            # report only
    python harvest_runs.py run7_indoor.npz --write                                      # also save samples
    python harvest_runs.py run6_indoor.npz --ignore run6_indoor.npz:117-131 --write     # skip 117-131 s (a sentence you got wrong)
    python harvest_runs.py run6_outdoor.npz --skip run6_outdoor.npz:2 --write           # you signed sentence 2 differently

Assumes the run follows the live-test protocol: the 20 recorded sentences in order, then
the held-out ones (gloss_data.RECORDED and HELD_OUT_ORDER). Look at the printed alignment
before using --write: a sentence you signed wrongly shows up as "unclear" or as strays, and
should be excluded with --ignore (a time range) or --skip (a sentence number, 1-based).
Samples go to Dataset/MP_Data_v2/idle/ (--out to change). After --write, run train.py.

A one-off stray can be a slip of the signer rather than a habit of the model, so a stray is only
used when the same pattern (previous gloss + stray, e.g. "thank_you" then "you") occurs at least
--min-count times across the runs given. Give it several runs. Only strays that stand clear of the real signs around them are used (--min-before / --min-after).
A window is one second long, so a stray right after or right before a real sign would put part
of that sign into an "idle" sample. The first version of this script did not check that, and the
retrained model then read head, ache, yes and good as idle. --write first deletes the stray_*.npy
files it made earlier, so re-running it never piles samples up.
"""
import argparse
import difflib
import os

import numpy as np
from tensorflow.keras.models import load_model

import gloss_data
from detector import GlossDetector
from features import N_MODEL_FEATURES, build_features
from gate import rest_gate
from stream import GlossStream
from utils import (ACTIONS_PATH, DATA_PATH, IDLE, MIN_WINDOW_FRAMES, MODEL_PATH, MU_PATH, SEQ_LEN,
                   SIGMA_PATH, WINDOW_SECONDS, resample_window)

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", help=".npz files saved by realtime.py --save-run")
ap.add_argument("--write", action="store_true", help="save the samples (default: only report)")
ap.add_argument("--out", default=os.path.join(DATA_PATH, IDLE), help="folder for the idle samples")
ap.add_argument("--ignore", action="append", default=[], metavar="RUN:T0-T1",
                help="ignore events between T0 and T1 seconds of that run (repeatable)")
ap.add_argument("--skip", action="append", default=[], metavar="RUN:N",
                help="that run's sentence number N (1-based) was not signed as planned (repeatable)")
ap.add_argument("--offsets", default="0,0.15", help="seconds after the stray's print at which a window ends")
ap.add_argument("--min-before", type=float, default=1.2,
                help="a stray closer than this many seconds after the previous gloss is not used (default 1.2)")
ap.add_argument("--min-after", type=float, default=1.0,
                help="a stray closer than this many seconds before the next gloss is not used (default 1.0)")
ap.add_argument("--min-count", type=int, default=2,
                help="a stray is only used if the same pattern (gloss before it + the stray) occurs at least this many "
                     "times in all the runs given; one-off strays may be slips of the signer (default 2)")
args = ap.parse_args()

SENTENCES = list(gloss_data.RECORDED) + list(gloss_data.HELD_OUT_ORDER)


def option(values, run):
    out = []
    for v in values:
        name, _, rest = v.rpartition(":")
        if os.path.basename(name) == os.path.basename(run):
            out.append(rest)
    return out


model = load_model(MODEL_PATH)
mu, sigma, actions = np.load(MU_PATH), np.load(SIGMA_PATH), [str(a) for a in np.load(ACTIONS_PATH)]
assert mu.shape[0] == N_MODEL_FEATURES, "run train.py first"
idle_i = actions.index(IDLE)
offsets = [float(x) for x in args.offsets.split(",")]


def window_ending(T, K, i):
    lo = int(np.searchsorted(T, T[i] - WINDOW_SECONDS))
    if i - lo + 1 < MIN_WINDOW_FRAMES or T[i] - T[lo] < 0.8 * WINDOW_SECONDS:
        return None
    return resample_window(T[lo:i + 1], K[lo:i + 1], SEQ_LEN)


def replay(T, K):
    """The live chain on saved frames: returns the glosses that ended up in the sentences."""
    idx, wins = [], []
    for i in range(len(T)):
        w = window_ending(T, K, i)
        if w is not None:
            idx.append(i); wins.append(w)
    probs = model.predict(np.array([((build_features(w) - mu) / sigma).astype("float32") for w in wins]),
                          batch_size=256, verbose=0)
    detector, stream, kept = GlossDetector(actions, idle=IDLE), GlossStream(max_items=10 ** 6), []
    for i, w, p in zip(idx, wins, probs):
        gated = rest_gate(w)[0]
        for label, prob in detector.update(np.eye(len(actions))[idle_i] if gated else p, T[i]):
            if stream.push(label, T[i])[0]:
                kept.append((float(T[i]), label, float(prob)))
    return kept


if args.write and os.path.isdir(args.out):
    old = [f for f in os.listdir(args.out) if f.startswith("stray_") and f.endswith(".npy")]
    for f in old:
        os.remove(os.path.join(args.out, f))
    if old:
        print(f"removed {len(old)} stray_*.npy files from an earlier harvest")
if args.write and os.path.isdir(args.out):
    old = [f for f in os.listdir(args.out) if f.startswith("stray_") and f.endswith(".npy")]
    for f in old:
        os.remove(os.path.join(args.out, f))
    if old:
        print(f"removed {len(old)} stray_*.npy files from an earlier harvest")

plans = []                                              # one entry per run, filled by the first pass
for run in args.runs:
    data = np.load(run)
    T, K = data["stamps"], data["keypoints"]
    name = os.path.splitext(os.path.basename(run))[0]
    events = replay(T, K)
    for r in option(args.ignore, run):
        a, b = (float(x) for x in r.split("-"))
        events = [e for e in events if not a <= e[0] <= b]
    skip = {int(x) for x in option(args.skip, run)}
    exp_tokens = [g for n, s in enumerate(SENTENCES, 1) if n not in skip for g in s.split()]
    det_tokens = [e[1] for e in events]
    ops = difflib.SequenceMatcher(None, exp_tokens, det_tokens, autojunk=False).get_opcodes()
    matched = sum(i2 - i1 for tag, i1, i2, _, _ in ops if tag == "equal")
    strays = [j for tag, _, _, j1, j2 in ops if tag == "insert" for j in range(j1, j2)]
    unclear = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in ops if tag == "replace"]
    missed = [exp_tokens[i] for tag, i1, i2, _, _ in ops if tag == "delete" for i in range(i1, i2)]
    print(f"\n{run}: {len(events)} glosses found, {matched} of {len(exp_tokens)} signed glosses matched, "
          f"{len(strays)} strays, {len(missed)} missed, {len(unclear)} unclear stretches")
    for j in strays:
        t, label, prob = events[j]
        before = det_tokens[j - 1] if j else "-"
        after = det_tokens[j + 1] if j + 1 < len(det_tokens) else "-"
        print(f"   stray  {t:6.1f}s  {label:<10s} ({prob:.2f})   between {before!r} and {after!r}")
    for i1, i2, j1, j2 in unclear:
        print(f"   unclear: signed {exp_tokens[i1:i2]} / found {det_tokens[j1:j2]}   "
              f"({events[j1][0]:.0f}-{events[j2 - 1][0]:.0f}s) -> not harvested; check it, use --ignore/--skip if you signed it differently")
    if missed:
        print(f"   missed (signed, not found): {missed}")
    plans.append((name, T, K, events, strays))

pattern_count = {}                                      # (gloss before, stray) -> how often over all runs
for name, T, K, events, strays in plans:
    for j in strays:
        key = (events[j - 1][1] if j else "-", events[j][1])
        pattern_count[key] = pattern_count.get(key, 0) + 1

total = 0
for name, T, K, events, strays in plans:
    n_saved = 0
    print(f"\n{name}:")
    for j in strays:
        t, label, prob = events[j]
        prev = events[j - 1][1] if j else "-"
        before = t - events[j - 1][0] if j else 1e9
        after = events[j + 1][0] - t if j + 1 < len(events) else 1e9
        if pattern_count[(prev, label)] < args.min_count:
            print(f"   not used: {label} after {prev!r} at {t:.1f}s happened only {pattern_count[(prev, label)]}x (maybe a slip)")
            continue
        if before < args.min_before or after < args.min_after:
            print(f"   not used: {label} at {t:.1f}s is {before:.1f}s after / {after:.1f}s before a real sign")
            continue
        for m, off in enumerate(offsets):
            i = min(int(np.searchsorted(T, t + off)), len(T) - 1)
            w = window_ending(T, K, i)
            if w is None:
                continue
            n_saved += 1
            if args.write:
                os.makedirs(args.out, exist_ok=True)
                np.save(os.path.join(args.out, f"stray_{name}_{int(round(t * 10))}_{m}.npy"), w.astype(np.float32))
    total += n_saved
    print(f"   {n_saved} idle windows {'saved to ' + args.out if args.write else 'would be saved (add --write)'}")
print(f"\n{total} idle samples in total" + ("" if args.write else " (nothing written yet)"))
