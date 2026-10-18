"""Train the sign classifier and save everything real-time testing needs.

Usage:
    python train.py

Saves to ./artifacts: sign_model.keras, mu.npy, sigma.npy, actions.npy,
confusion_matrix.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Bidirectional, Dense, Dropout,
                                     GaussianNoise, Input, LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from utils import (ACTIONS, ACTIONS_PATH, ARTIFACTS, DATA_PATH, MODEL_PATH,
                   MU_PATH, N_FEATURES, SEQ_LEN, SIGMA_PATH)

SEED = 42
os.makedirs(ARTIFACTS, exist_ok=True)

# ------------------------------------------------------------------ load data
X, y = [], []
for idx, action in enumerate(ACTIONS):
    folder = os.path.join(DATA_PATH, action)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"No data for '{action}'. Run: python record.py {action}")
    files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    for f in files:
        sample = np.load(os.path.join(folder, f))
        if sample.shape != (SEQ_LEN, N_FEATURES):
            print(f"skipping {action}/{f}: bad shape {sample.shape}")
            continue
        X.append(sample)
        y.append(idx)
    print(f"{action:>10}: {y.count(idx)} samples")

X = np.array(X, dtype=np.float32)
y = np.array(y)
if min(np.bincount(y, minlength=len(ACTIONS))) < 6:
    raise ValueError("Need at least ~6 samples per class (30+ recommended).")

# ----------------------------------------------------- split (stratified, 70/15/15)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=SEED)
print(f"train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

# Normalisation statistics from the TRAIN split only, one value per feature
# (shape (258,)), so they apply to any 30-frame window at inference time.
mu = X_train.mean(axis=(0, 1))
sigma = np.maximum(X_train.std(axis=(0, 1)), 1e-3)
X_train, X_val, X_test = [(a - mu) / sigma for a in (X_train, X_val, X_test)]

n_classes = len(ACTIONS)
Y_train = to_categorical(y_train, n_classes)
Y_val = to_categorical(y_val, n_classes)

# ---------------------------------------------------------------------- model
model = Sequential([
    Input(shape=(SEQ_LEN, N_FEATURES)),
    GaussianNoise(0.05),                       # light augmentation, active only in training
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(n_classes, activation="softmax"),
])
model.compile(optimizer=Adam(5e-4), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
          epochs=150, batch_size=8,
          callbacks=[EarlyStopping(monitor="val_loss", patience=20,
                                   restore_best_weights=True)])

# ----------------------------------------------------------------- save
model.save(MODEL_PATH)
np.save(MU_PATH, mu)
np.save(SIGMA_PATH, sigma)
np.save(ACTIONS_PATH, np.array(ACTIONS))
print(f"Saved model and normalisation to ./{ARTIFACTS}")

# ------------------------------------------------------------- evaluation
probs = model.predict(X_test, verbose=0)
y_pred = probs.argmax(axis=1)
labels = list(range(n_classes))
print(classification_report(y_test, y_pred, labels=labels,
                            target_names=ACTIONS, zero_division=0))
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.xticks(labels, ACTIONS, rotation=45)
plt.yticks(labels, ACTIONS)
for i in range(n_classes):
    for j in range(n_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion matrix ({n_classes} classes)")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS, "confusion_matrix.png"), dpi=120)
print("Saved confusion_matrix.png")
