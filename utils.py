"""Shared settings and helpers for recording, training and real-time testing.

Everything that must match between the three stages lives here, so the
stages cannot drift apart.
"""
import os

import cv2
import numpy as np
import mediapipe as mp

# ---------------------------------------------------------------- vocabulary
# Stage 1: one gloss. Stage 2: add more words to this list, record each one
# with record.py, then retrain. Nothing else needs to change.
GLOSSES = ["hello","name","head","ache","stomach"]

# A one-gloss model needs a "not signing" class, otherwise softmax always
# outputs 1.0 for "hello" and the model can never say "nothing is happening".
IDLE = "idle"
ACTIONS = GLOSSES + [IDLE]

# --------------------------------------------------------------------- shape
SEQ_LEN = 30                       # frames per sample (~1 second)
N_FEATURES = 33 * 4 + 21 * 3 * 2   # pose (x,y,z,vis) + left hand + right hand = 258

# --------------------------------------------------------------------- paths
DATA_PATH = os.path.join("Dataset", "MP_Data_v2")   # new folder: do not mix with old data
ARTIFACTS = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS, "sign_model.keras")
MU_PATH = os.path.join(ARTIFACTS, "mu.npy")
SIGMA_PATH = os.path.join(ARTIFACTS, "sigma.npy")
ACTIONS_PATH = os.path.join(ARTIFACTS, "actions.npy")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def make_holistic():
    return mp_holistic.Holistic(min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    """Draw pose and hands only (the face mesh is not used by the model)."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


def extract_keypoints(results):
    """Return a (258,) vector: pose + left hand + right hand.

    Coordinates are made position/size independent: shifted so the midpoint
    between the shoulders is the origin and divided by the shoulder width.
    Missing hands are all zeros (same at training and at inference).
    """
    if not results.pose_landmarks:
        return np.zeros(N_FEATURES, dtype=np.float32)

    pose = np.array([[p.x, p.y, p.z, p.visibility]
                     for p in results.pose_landmarks.landmark], dtype=np.float32)
    left_sh, right_sh = pose[11, :3], pose[12, :3]
    origin = (left_sh + right_sh) / 2.0
    scale = float(np.linalg.norm(left_sh[:2] - right_sh[:2]))
    if scale < 1e-3:
        return np.zeros(N_FEATURES, dtype=np.float32)

    pose[:, :3] = (pose[:, :3] - origin) / scale

    def hand(h):
        if h is None:
            return np.zeros(21 * 3, dtype=np.float32)
        a = np.array([[p.x, p.y, p.z] for p in h.landmark], dtype=np.float32)
        return ((a - origin) / scale).flatten()

    return np.concatenate([pose.flatten(),
                           hand(results.left_hand_landmarks),
                           hand(results.right_hand_landmarks)])
