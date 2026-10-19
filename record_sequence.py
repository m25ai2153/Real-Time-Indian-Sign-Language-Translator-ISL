"""Record signs with on-screen cues, alone or back to back with no rest in between.
Every sign gets its own duration, so a 0.5 s sign and a 1.2 s sign can be recorded
in the same sequence.

Usage:
    python record_sequence.py need:1.2                       # one sign on its own (30 rounds)
    python record_sequence.py head:0.5 ache:0.5              # two short signs, no rest between them
    python record_sequence.py doctor:1.1 head:0.5 --rounds 20
    python record_sequence.py head ache --beat 0.6           # signs without ':seconds' use --beat
    python record_sequence.py head:0.5 ache:0.5 --practice   # rehearse the rhythm, saves nothing
    python record_sequence.py need:1.2 --min-hand 0.6        # skip samples where a hand was lost

The number after the colon is how long THAT sign takes when you sign it naturally.
It is not the model's window (WINDOW_SECONDS in utils.py), which is fixed: a short
sign sits inside the window with rest or its neighbours around it, a long sign
fills most of it. The signer is never asked to slow a short sign down.

How a round works: a "GET READY" countdown, then one cue per sign (shown for that
sign's duration), then "RELAX". Sign during the cue and flow straight into the next
one without returning to rest, as in a real conversation. Rehearse with --practice.

For every cue one training sample is cut out of the continuous recording, centred on
the middle of that cue (plus a small random offset), and saved in that gloss' folder:
  * solo_<n>.npy   when the round has a single sign (a clean example with rest around it)
  * seq_<n>.npy    when signs are back to back (the window holds the tail of the previous
                   sign and the start of the next one, exactly what the live recogniser sees)
train.py and inspect_data.py pick both up. The whole round is also stored in
Dataset/MP_Sequences so whole sentences can be tested offline later.

Tips:
  * Record each pair at more than one speed, e.g. head:0.5 ache:0.5 and head:0.8 ache:0.8.
  * Record several pairs and both orders, so a sign is seen with different neighbours.
  * Use idle (record.py idle) for rest, hands resting, and non-sign movements.
"""
import argparse
import os
import time

import cv2
import numpy as np

from utils import (DATA_PATH, GLOSSES, MIN_WINDOW_FRAMES, SEQ_LEN, WINDOW_SECONDS,
                   draw_landmarks, extract_keypoints, make_holistic,
                   mediapipe_detection, open_camera, resample_window)

COUNTDOWN_SECONDS = 2.0
SEQ_PATH = os.path.join(os.path.dirname(DATA_PATH) or ".", "MP_Sequences")
PALETTE = [(0, 120, 255), (0, 170, 80), (200, 60, 200), (0, 170, 200), (200, 140, 0)]


def parse_cues(tokens, default_beat, allowed):
    """['head:0.5', 'ache'] -> [('head', 0.5), ('ache', default_beat)]"""
    cues = []
    for tok in tokens:
        name, _, secs = tok.partition(":")
        if name not in allowed:
            raise ValueError(f"unknown sign '{name}' (known signs: {', '.join(allowed)})")
        try:
            dur = float(secs) if secs else float(default_beat)
        except ValueError:
            raise ValueError(f"'{tok}': the part after ':' must be a number of seconds")
        if not 0.2 <= dur <= 4.0:
            raise ValueError(f"'{tok}': seconds must be between 0.2 and 4.0")
        cues.append((name, dur))
    return cues


def next_seq_index(folder, prefix="seq"):
    """Next free number for <prefix>_<n>.npy files in a folder."""
    nums = []
    for f in os.listdir(folder):
        stem, ext = os.path.splitext(f)
        if ext == ".npy" and stem.startswith(prefix + "_") and stem[len(prefix) + 1:].isdigit():
            nums.append(int(stem[len(prefix) + 1:]))
    return max(nums) + 1 if nums else 0


def cut_samples(stamps, kps, t0, starts, durs, jitter, rng,
                window=WINDOW_SECONDS, seq_len=SEQ_LEN, min_frames=MIN_WINDOW_FRAMES):
    """Cut one window per cue out of a continuous recording.

    Cue i spans [t0 + starts[i], t0 + starts[i] + durs[i]]. The window is centred on the
    middle of the cue, shifted by a random fraction (+-jitter) of that cue's duration.
    Returns one entry per cue: None if the window had too few real frames, else a dict
    with the resampled (seq_len, 258) sample, the number of real frames and the fraction
    of frames in which a hand was tracked.
    """
    stamps = np.asarray(stamps, dtype=np.float64)
    kps = np.asarray(kps, dtype=np.float32)
    out = []
    for start, dur in zip(starts, durs):
        center = t0 + start + dur / 2 + rng.uniform(-jitter, jitter) * dur
        sel = (stamps >= center - window / 2) & (stamps <= center + window / 2)
        ts, ks = stamps[sel], kps[sel]
        # same readiness rule as realtime.py
        if len(ts) < min_frames or ts[-1] - ts[0] < 0.8 * window:
            out.append(None)
            continue
        out.append({"sample": resample_window(ts, ks, seq_len),
                    "n_real": len(ts),
                    "hand": float(np.mean(np.any(ks[:, 132:] != 0, axis=1)))})
    return out


def put_centered(image, text, y, scale, color, thickness):
    if not text:
        return
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max((image.shape[1] - tw) // 2, 5)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 4, cv2.LINE_AA)               # outline for readability
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def draw_cue(image, now, t0, t_end, starts, durs, names, round_no, rounds, practice):
    h, w = image.shape[:2]
    tag = "PRACTICE  " if practice else ""
    big_color = (255, 255, 255)
    rel = now - t0
    if now < t0:
        color, frac = (60, 60, 60), 0.0
        head = f"{tag}round {round_no}/{rounds} - GET READY"
        big, sub = str(int(t0 - now) + 1), f"first sign: {names[0].upper()}"
        big_color = (0, 255, 255)
    elif now < t_end:
        i = max(k for k, s in enumerate(starts) if s <= rel)
        frac = min((rel - starts[i]) / durs[i], 1.0)
        color = PALETTE[i % len(PALETTE)]
        head = f"{tag}round {round_no}/{rounds} - sign {i + 1}/{len(names)} ({durs[i]:.1f}s)"
        big = names[i].upper()
        sub = ("then: " + names[i + 1].upper()) if i + 1 < len(names) else "then: relax"
    else:
        color, frac = (60, 60, 60), 1.0
        head = f"{tag}round {round_no}/{rounds} - relax"
        big, sub = "RELAX", ""
    cv2.rectangle(image, (0, 0), (w, 50), color, -1)
    cv2.putText(image, head, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    put_centered(image, big, h // 2 + 30, 2.4, big_color, 5)
    put_centered(image, sub, h // 2 + 80, 0.9, (255, 255, 255), 2)
    cv2.rectangle(image, (0, h - 14), (int(w * frac), h), color, -1)   # progress of this cue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("signs", nargs="+",
                        help="signs in the order they are signed, each optionally with its own "
                             "duration in seconds, e.g.  head:0.5 ache:0.5  or  need:1.2")
    parser.add_argument("--rounds", type=int, default=None,
                        help="how many times to repeat (default 30 for one sign, 15 for a sequence)")
    parser.add_argument("--beat", type=float, default=0.7,
                        help="duration for signs given without ':seconds'")
    parser.add_argument("--jitter", type=float, default=0.15,
                        help="random window offset, as a fraction of the sign's duration")
    parser.add_argument("--min-hand", type=float, default=0.0,
                        help="skip a sample if a hand was tracked in less than this fraction of frames")
    parser.add_argument("--practice", action="store_true", help="rehearse only, save nothing")
    args = parser.parse_args()
    try:
        cues = parse_cues(args.signs, args.beat, GLOSSES)
    except ValueError as e:
        parser.error(str(e))

    names = [c[0] for c in cues]
    durs = [c[1] for c in cues]
    starts = list(np.concatenate([[0.0], np.cumsum(durs)[:-1]]))
    total = float(sum(durs))
    solo = len(cues) == 1
    prefix = "solo" if solo else "seq"
    rounds = args.rounds or (30 if solo else 15)
    rng = np.random.default_rng()
    saved = {g: 0 for g in names}

    print("Cues: " + "  ->  ".join(f"{n} ({d:.1f}s)" for n, d in cues) +
          f"   |   model window {WINDOW_SECONDS}s, {SEQ_LEN} steps   |   {rounds} rounds. Press 'q' to stop.")
    longer = [f"{n} ({d:.1f}s)" for n, d in cues if d > WINDOW_SECONDS]
    if longer:
        print(f"NOTE: longer than the {WINDOW_SECONDS}s window: {', '.join(longer)}. The window will see "
              f"only the middle of those signs. Increase WINDOW_SECONDS in utils.py if that matters.")

    cap = open_camera()
    try:
        with make_holistic() as holistic:
            for r in range(1, rounds + 1):
                stamps, kps = [], []
                t_ready = time.perf_counter()
                t0 = t_ready + COUNTDOWN_SECONDS               # first cue starts here
                t_end = t0 + total
                t_stop = t_end + WINDOW_SECONDS                # keep filming a little after the last cue
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        return
                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)
                    now = time.perf_counter()
                    if now >= t_stop:
                        break
                    stamps.append(now)
                    kps.append(extract_keypoints(results))
                    draw_cue(image, now, t0, t_end, starts, durs, names, r, rounds, args.practice)
                    cv2.imshow("Record sequence", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Stopped; the unfinished round was discarded.")
                        return

                if args.practice:
                    print(f"round {r}: practice, nothing saved")
                    continue

                cuts = cut_samples(stamps, kps, t0, starts, durs, args.jitter, rng,
                                   window=WINDOW_SECONDS, seq_len=SEQ_LEN)
                for gloss, cut in zip(names, cuts):
                    if cut is None:
                        print(f"round {r}: {gloss}: too few real frames in the window - skipped "
                              f"(close other apps / lower camera resolution)")
                        continue
                    if cut["hand"] < args.min_hand:
                        print(f"round {r}: {gloss}: hand tracked in only {cut['hand']:.0%} of frames "
                              f"(< {args.min_hand:.0%}) - skipped")
                        continue
                    folder = os.path.join(DATA_PATH, gloss)
                    os.makedirs(folder, exist_ok=True)
                    idx = next_seq_index(folder, prefix)
                    np.save(os.path.join(folder, f"{prefix}_{idx}.npy"), cut["sample"])
                    saved[gloss] += 1
                    print(f"round {r}: saved {gloss}/{prefix}_{idx}.npy  ({cut['n_real']} real frames "
                          f"= ~{cut['n_real'] / WINDOW_SECONDS:.0f} fps, hand tracked {cut['hand']:.0%})")

                # whole round, for testing complete sentences offline later (times relative to the first cue)
                seq_dir = os.path.join(SEQ_PATH, "_".join(names))
                os.makedirs(seq_dir, exist_ok=True)
                k = len([f for f in os.listdir(seq_dir) if f.endswith(".npz")])
                np.savez_compressed(os.path.join(seq_dir, f"{k}.npz"),
                                    stamps=np.asarray(stamps) - t0,
                                    keypoints=np.asarray(kps, dtype=np.float32),
                                    glosses=np.array(names),
                                    durations=np.array(durs),
                                    window=WINDOW_SECONDS)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if not args.practice:
            print("Saved this run: " + ", ".join(f"{g} {c}" for g, c in saved.items()))


if __name__ == "__main__":
    main()