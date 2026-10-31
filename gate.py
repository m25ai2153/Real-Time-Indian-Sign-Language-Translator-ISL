"""Rest gate (numpy only): decide from the raw landmarks whether a window can be a sign at all.

The classifier is only as good as its idle examples, and in a live test a still
resting posture (hands in the lap, no hand visible) was classified as a sign for
seconds at a time. Two facts hold for every sign in the data and are cheap to check:
  * a sign shows at least one hand, so a window with (almost) no tracked hand is rest;
  * a sign moves an arm, so a window in which both wrists hardly move is rest;
  * a sign is made in front of the body, so a window in which both wrists stay down in the
    lap is rest, even when the camera can see the hands there (run6: hands resting in the
    lap with both hands tracked were read as the sign "I" for seconds).
In your own recordings the quietest sign windows moved ~0.3 shoulder widths, a still
resting window 0.03-0.10, so 0.15 sits in the gap.
"""
import numpy as np

MIN_HAND_FRACTION = 0.10      # share of window steps with a tracked hand
MIN_WRIST_MOTION = 0.0        # 0 = motion check off: a held sign like "good" barely moves the wrist
REST_BELOW = 1.2              # shoulder widths below the shoulder line: both wrists lower than this all window = rest
LEFT_WRIST, RIGHT_WRIST = 15, 16


def window_motion(window):
    """Largest x/y extent (shoulder widths) covered by either pose wrist inside one (T, 258) window."""
    w = np.asarray(window, dtype=np.float32)
    best = 0.0
    for idx in (LEFT_WRIST, RIGHT_WRIST):
        xy = w[:, 4 * idx:4 * idx + 2]
        best = max(best, float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0))))
    return best


def highest_wrist(window):
    """Smallest y (= highest point, y grows downwards) reached by either pose wrist in the window."""
    w = np.asarray(window, dtype=np.float32)
    return float(min(w[:, 4 * LEFT_WRIST + 1].min(), w[:, 4 * RIGHT_WRIST + 1].min()))


def rest_gate(window, min_hand=MIN_HAND_FRACTION, min_motion=MIN_WRIST_MOTION):
    """Returns (is_rest, hand_fraction, motion) for one (T, 258) window of stored keypoints."""
    w = np.asarray(window, dtype=np.float32)
    hand = float(np.mean(np.any(w[:, 132:258] != 0, axis=1)))
    motion = window_motion(w)
    low = highest_wrist(w) > REST_BELOW
    return (hand < min_hand or motion < min_motion or low), hand, motion
