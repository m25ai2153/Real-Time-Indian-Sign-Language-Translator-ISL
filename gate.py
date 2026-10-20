"""Rest gate (numpy only): decide from the raw landmarks whether a window can be a sign at all.

The classifier is only as good as its idle examples, and in a live test a still
resting posture (hands in the lap, no hand visible) was classified as a sign for
seconds at a time. Two facts hold for every sign in the data and are cheap to check:
  * a sign shows at least one hand, so a window with (almost) no tracked hand is rest;
  * a sign moves an arm, so a window in which both wrists hardly move is rest.
In your own recordings the quietest sign windows moved ~0.3 shoulder widths, a still
resting window 0.03-0.10, so 0.15 sits in the gap.
"""
import numpy as np

MIN_HAND_FRACTION = 0.10      # share of window steps with a tracked hand
MIN_WRIST_MOTION = 0.15       # in shoulder widths: biggest x/y range of either wrist
LEFT_WRIST, RIGHT_WRIST = 15, 16


def window_motion(window):
    """Largest x/y extent (shoulder widths) covered by either pose wrist inside one (T, 258) window."""
    w = np.asarray(window, dtype=np.float32)
    best = 0.0
    for idx in (LEFT_WRIST, RIGHT_WRIST):
        xy = w[:, 4 * idx:4 * idx + 2]
        best = max(best, float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0))))
    return best


def rest_gate(window, min_hand=MIN_HAND_FRACTION, min_motion=MIN_WRIST_MOTION):
    """Returns (is_rest, hand_fraction, motion) for one (T, 258) window of stored keypoints."""
    w = np.asarray(window, dtype=np.float32)
    hand = float(np.mean(np.any(w[:, 132:258] != 0, axis=1)))
    motion = window_motion(w)
    return (hand < min_hand or motion < min_motion), hand, motion