"""Feature building and augmentation (numpy only: no TensorFlow / MediaPipe needed).

record.py stores 258 values per frame (pose + both hands in body coordinates).
Those stored samples are turned into the model's input here, at train time and
at live time, so you can change the features without re-recording anything.

What the model sees per frame (333 values):
  * upper-body pose only (23 landmarks, x/y/z): legs and the per-landmark
    visibility values were noise for signing
  * both hands in body coordinates (where the hand is)
  * both hands relative to their own wrist and scaled by hand size (what the
    hand SHAPE is). This is what separates "open palm" from "index finger only"
  * a 0/1 flag per hand: was it really tracked in this frame?
  * 5 finger-extension ratios per hand (fingertip-to-wrist distance divided by
    knuckle-to-wrist distance): high = finger extended, low = curled

MediaPipe often loses a hand for a few frames while it moves fast or passes
the face. Short dropouts (up to ~0.13 s) are filled from the nearest tracked
frame so the model still sees a hand shape; the flag keeps the true tracking
state visible. The same code runs when training and live, so they match.
"""
import numpy as np

STORED_FEATURES = 258        # per frame, as saved by record.py
POSE_LANDMARKS_KEPT = 23     # nose, eyes, ears, mouth, shoulders, elbows, wrists, hand points
TIPS = [4, 8, 12, 16, 20]    # thumb, index, middle, ring, pinky tips
KNUCKLES = [2, 5, 9, 13, 17]
LEFT_HAND = slice(132, 195)
RIGHT_HAND = slice(195, 258)


def fill_hand_gaps(window, max_gap=8, max_edge=4):
    """Fill short tracking dropouts of each hand in one (T, 258) window.

    A gap between two tracked frames is filled (from the nearer one) if it is at
    most `max_gap` steps long; a gap at the start or end of the window if it is
    at most `max_edge` steps from the nearest tracked frame. Longer gaps stay
    empty, because the hand really left the picture.
    """
    out = np.array(window, dtype=np.float32, copy=True)
    for sl in (LEFT_HAND, RIGHT_HAND):
        present = np.any(out[:, sl] != 0, axis=1)
        if present.all() or not present.any():
            continue
        src = out[:, sl].copy()
        for k in np.flatnonzero(~present):
            before = np.flatnonzero(present[:k])
            after = np.flatnonzero(present[k + 1:])
            p = int(before[-1]) if len(before) else None
            n = k + 1 + int(after[0]) if len(after) else None
            if p is not None and n is not None:
                if n - p - 1 > max_gap:
                    continue
                j = p if (k - p) <= (n - k) else n
            elif p is not None:
                if k - p > max_edge:
                    continue
                j = p
            else:
                if n - k > max_edge:
                    continue
                j = n
            out[k, sl] = src[j]
    return out


def _hand_shape(h):
    """h: (n, 21, 3) hand points. Returns wrist-relative scaled points (n, 21, 3)
    and finger-extension ratios (n, 5). Missing hands give zeros."""
    wrist = h[:, :1, :]
    size = np.linalg.norm(h[:, 9, :] - h[:, 0, :], axis=1)      # wrist -> middle knuckle
    present = np.any(h != 0, axis=(1, 2)) & (size > 1e-4)
    safe = np.where(present, size, 1.0)
    rel = (h - wrist) / safe[:, None, None]
    rel[~present] = 0.0
    d_tip = np.linalg.norm(h[:, TIPS, :] - wrist, axis=2)
    d_knuckle = np.linalg.norm(h[:, KNUCKLES, :] - wrist, axis=2)
    ext = d_tip / np.maximum(d_knuckle, 1e-4)
    ext[~present] = 0.0
    return rel, ext


def build_features(window, fill_gaps=True):
    """(..., T, 258) stored keypoints -> (..., T, N_MODEL_FEATURES).

    Accepts one window (T, 258) or a batch (N, T, 258).
    """
    w = np.asarray(window, dtype=np.float32)
    lead, t = w.shape[:-2], w.shape[-2]
    w3 = w.reshape(-1, t, STORED_FEATURES)

    # which hands were really tracked (before any filling)
    tracked = np.stack([np.any(w3[..., LEFT_HAND] != 0, axis=-1),
                        np.any(w3[..., RIGHT_HAND] != 0, axis=-1)], axis=-1).astype(np.float32)
    if fill_gaps:
        w3 = np.stack([fill_hand_gaps(x) for x in w3])

    f = w3.reshape(-1, STORED_FEATURES)
    n = f.shape[0]
    pose = f[:, :132].reshape(n, 33, 4)[:, :POSE_LANDMARKS_KEPT, :3].reshape(n, -1)
    lh = f[:, LEFT_HAND].reshape(n, 21, 3)
    rh = f[:, RIGHT_HAND].reshape(n, 21, 3)
    lh_rel, lh_ext = _hand_shape(lh)
    rh_rel, rh_ext = _hand_shape(rh)
    out = np.concatenate([pose, lh.reshape(n, -1), rh.reshape(n, -1),
                          lh_rel.reshape(n, -1), rh_rel.reshape(n, -1),
                          tracked.reshape(n, 2), lh_ext, rh_ext], axis=1)
    return out.reshape(*lead, t, -1).astype(np.float32)


N_MODEL_FEATURES = build_features(np.zeros((1, STORED_FEATURES))).shape[-1]


def augment_window(window, rng, max_shift_frac=0.17, stretch=(0.7, 1.5)):
    """Randomly shift (+-max_shift_frac of the window) and speed up / slow down one
    (T, F) window. The sign appears earlier/later and faster/slower inside the window,
    which is what happens live: the window slides over the sign, and one signer is
    faster or slower than another. Edge frames are repeated to fill any gap.
    stretch > 1 plays the window faster (the sign looks shorter), < 1 slower."""
    t = len(window)
    shift = rng.uniform(-max_shift_frac, max_shift_frac) * t
    factor = rng.uniform(*stretch)
    pos = (np.arange(t) - (t - 1) / 2.0) * factor + (t - 1) / 2.0 + shift
    idx = np.clip(np.rint(pos), 0, t - 1).astype(int)
    return window[idx]


# ----------------------------------------------------------- left-hand decorrelation
# In a live test the signer rested the LEFT hand on the desk, in view, while signing with
# the right one. In the training data a visible left hand mostly meant "idle" (the signs
# are nearly all right-handed), so the model called nearly every sign idle. The helpers
# below break that shortcut: a resting left hand is added to right-handed training
# samples, and removed from some idle samples, so its presence says nothing about the class.
LEFT_ARM_POSE = (13, 15, 17, 19, 21)       # left elbow, wrist, pinky, index, thumb in the pose block
LEFT_ARM_COLS = np.concatenate([np.arange(4 * i, 4 * i + 4) for i in LEFT_ARM_POSE])


def left_tracked_fraction(window):
    """Share of frames in which the left hand was tracked, for (T, 258) or (N, T, 258)."""
    w = np.asarray(window)
    return np.any(w[..., LEFT_HAND] != 0, axis=-1).mean(axis=-1)


def set_left(window, hand, arm):
    """Copy of a (T, 258) window with the left hand block and left arm pose replaced."""
    out = np.array(window, dtype=np.float32, copy=True)
    out[:, LEFT_HAND] = hand
    out[:, LEFT_ARM_COLS] = arm
    return out


def left_arm_at_rest(X):
    """Median left-arm pose (20 values) over all frames where the left hand was NOT tracked
    (hand out of view, in the lap). Used to 'hide' a resting left hand."""
    f = np.asarray(X, dtype=np.float32).reshape(-1, STORED_FEATURES)
    lost = ~np.any(f[:, LEFT_HAND] != 0, axis=1)
    return np.median(f[lost][:, LEFT_ARM_COLS], axis=0) if lost.any() else None


class LeftPool:
    """Real frames of a resting left hand (hand block + left arm pose), taken from a --save-run
    recording in which the left hand stayed in view. Chunks are cut from long stretches in
    which the left hand was tracked all the time."""

    def __init__(self, frames, starts, lengths):
        self.frames = np.asarray(frames, dtype=np.float32)
        self.starts = np.asarray(starts, dtype=int)
        self.lengths = np.asarray(lengths, dtype=int)

    @classmethod
    def from_runs(cls, runs, min_len=32):
        frames, starts, lengths, pos = [], [], [], 0
        for kp in runs:
            kp = np.asarray(kp, dtype=np.float32)
            ok = np.any(kp[:, LEFT_HAND] != 0, axis=1)
            i = 0
            while i < len(ok):
                if not ok[i]:
                    i += 1
                    continue
                j = i
                while j < len(ok) and ok[j]:
                    j += 1
                if j - i >= min_len:
                    frames.append(kp[i:j])
                    starts.append(pos)
                    lengths.append(j - i)
                    pos += j - i
                i = j
        if not frames:
            raise ValueError("no stretch of at least %d frames with the left hand tracked" % min_len)
        return cls(np.concatenate(frames), starts, lengths)

    def save(self, path):
        np.savez_compressed(path, frames=self.frames, starts=self.starts, lengths=self.lengths)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["frames"], z["starts"], z["lengths"])

    def chunk(self, rng, t):
        """A random (t, 258) stretch of resting-left-hand frames."""
        usable = np.flatnonzero(self.lengths >= t)
        k = int(rng.choice(usable)) if len(usable) else int(rng.integers(len(self.lengths)))
        n = int(self.lengths[k])
        start = int(self.starts[k]) + (int(rng.integers(0, n - t + 1)) if n >= t else 0)
        frames = self.frames[start:start + t]
        if len(frames) < t:                                  # short stretch: repeat the last frame
            frames = np.concatenate([frames, np.repeat(frames[-1:], t - len(frames), axis=0)])
        return frames


def inject_left_hand(window, pool, rng):
    """Add a resting left hand (and left arm) to a window that has none."""
    chunk = pool.chunk(rng, len(window))
    return set_left(window, chunk[:, LEFT_HAND], chunk[:, LEFT_ARM_COLS])


def hide_left_hand(window, rest_arm):
    """Remove the left hand from a window and put the left arm back in the lap."""
    t = len(window)
    return set_left(window, np.zeros((t, LEFT_HAND.stop - LEFT_HAND.start), np.float32),
                    np.tile(rest_arm, (t, 1)))


def add_augmented(X, y, copies, seed=0, left_pool=None, rest_arm=None, idle_idx=None,
                  p_inject=0.5, p_hide=0.4):
    """Return X, y plus `copies` augmented versions of every sample.

    Every copy is shifted / stretched in time. With a `left_pool`, a resting left hand is
    added (probability p_inject) to samples that have no left hand; with `rest_arm` and
    `idle_idx`, the left hand is removed (probability p_hide) from idle samples that have one.
    """
    rng = np.random.default_rng(seed)
    lfrac = left_tracked_fraction(X)
    xs, ys = [X], [y]
    for _ in range(copies):
        batch = []
        for w, label, lf in zip(X, y, lfrac):
            w = augment_window(w, rng)
            if left_pool is not None and lf < 0.3 and rng.random() < p_inject:
                w = inject_left_hand(w, left_pool, rng)
            elif (rest_arm is not None and idle_idx is not None and label == idle_idx
                  and lf >= 0.3 and rng.random() < p_hide):
                w = hide_left_hand(w, rest_arm)
            batch.append(w)
        xs.append(np.stack(batch))
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)