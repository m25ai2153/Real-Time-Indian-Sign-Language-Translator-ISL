"""The demo screen: video top-left, glosses top-right, English translation along the bottom.

    +-----------------------------+----------------+
    |                             |  GLOSSES       |
    |        VIDEO                |  I, head, ache |
    |                             |                |
    +-----------------------------+----------------+
    |  TRANSLATION                                 |
    |  I have a headache, need a doctor.           |
    +----------------------------------------------+

compose() only draws; it holds no state, so it can be tested with a still image.
"""
import cv2
import numpy as np

VIDEO_W, VIDEO_H = 640, 480
SIDE_W = 420
BOTTOM_H = 230
W, H = VIDEO_W + SIDE_W, VIDEO_H + BOTTOM_H

BG = (32, 32, 32)
LINE = (235, 235, 235)
ORANGE = (16, 117, 245)        # BGR
GREEN = (90, 220, 90)
GREY = (150, 150, 150)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def wrap(text, scale, thickness, max_w):
    """Split text into lines no wider than max_w pixels."""
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if cv2.getTextSize(trial, FONT, scale, thickness)[0][0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def header(canvas, text, x, y):
    cv2.putText(canvas, text, (x, y), FONT, 0.8, ORANGE, 2, cv2.LINE_AA)
    width = cv2.getTextSize(text, FONT, 0.8, 2)[0][0]
    cv2.line(canvas, (x, y + 6), (x + width, y + 6), ORANGE, 2)


def compose(video, glosses, translation, pending=False, history=(), info="", probs=None,
            actions=None, saved=False):
    """Return the full screen as a BGR image.

    video: camera image (any size). glosses: list of strings. translation: the English text
    for the current glosses. pending: a new translation is being computed. history: earlier
    sentences (oldest first). info: small status text (fps). probs/actions: optional bars.
    """
    canvas = np.full((H, W, 3), BG, np.uint8)
    img = cv2.resize(video, (VIDEO_W, VIDEO_H))
    if probs is not None and actions is not None:
        for i, v in enumerate(probs):
            cv2.rectangle(img, (0, 50 + i * 26), (int(v * 160), 70 + i * 26), ORANGE, -1)
            cv2.putText(img, str(actions[i]), (4, 66 + i * 26), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    canvas[:VIDEO_H, :VIDEO_W] = img
    if info:
        cv2.putText(canvas, info, (8, 22), FONT, 0.5, WHITE, 1, cv2.LINE_AA)

    # glosses panel (top right)
    x0 = VIDEO_W + 18
    header(canvas, "GLOSSES", x0, 42)
    text = ", ".join(g.replace("_", " ") for g in glosses)
    scale = 1.0 if len(glosses) <= 8 else 0.8
    for i, line in enumerate(wrap(text, scale, 2, SIDE_W - 36)[:9]):
        cv2.putText(canvas, line, (x0, 92 + i * 40), FONT, scale, WHITE, 2, cv2.LINE_AA)

    # translation panel (bottom)
    header(canvas, "TRANSLATION", 18, VIDEO_H + 40)
    shown = translation + (" ..." if pending and translation else "")
    if pending and not translation and glosses:
        shown = "..."
    big = 1.15 if len(shown) <= 60 else 0.9
    lines = wrap(shown, big, 2, W - 40)[:3]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (18, VIDEO_H + 90 + i * 40), FONT, big, GREEN, 2, cv2.LINE_AA)
    y = H - 12                                           # earlier sentences, small, bottom up
    for old in list(history)[-2:][::-1]:
        cv2.putText(canvas, old[:110], (18, y), FONT, 0.5, GREY, 1, cv2.LINE_AA)
        y -= 22

    # frame lines as in the sketch
    cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), LINE, 2)
    cv2.line(canvas, (VIDEO_W, 0), (VIDEO_W, VIDEO_H), LINE, 2)
    cv2.line(canvas, (0, VIDEO_H), (W - 1, VIDEO_H), LINE, 2)
    if saved:
        cv2.putText(canvas, "SAVED", (VIDEO_W // 2 - 60, VIDEO_H // 2), FONT, 1.6, GREEN, 3, cv2.LINE_AA)
    return canvas
