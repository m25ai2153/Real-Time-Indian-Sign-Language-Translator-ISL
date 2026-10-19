"""Train the sign classifier and save everything real-time testing needs.

Usage:
    python train.py

Saves to ./artifacts: sign_model.keras, mu.npy, sigma.npy, actions.npy,
confusion_matrix.png

Compared with the first version:
  * hand-shape features (features.py) so open palm vs one finger is learnable
  * temporal shift / speed augmentation of the TRAIN split only (the test and
    validation samples stay untouched, so the scores stay honest)
  * balanced class weights so a class with fewer samples (e.g. idle) is not ignored
  * a list of the most confused class pairs at the end
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Bidirectional, Dense, Dropout,
                                     GaussianNoise, Input, LSTM)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from features import N_MODEL_FEATURES, STORED_FEATURES, add_augmented, build_features
from utils import (ACTIONS, ACTIONS_PATH, ARTIFACTS, DATA_PATH, MODEL_PATH,
                   MU_PATH, N_FEATURES, SEQ_LEN, SIGMA_PATH)

SEED = 42
AUG_COPIES = 8          # augmented copies per training sample (0 = off)
assert STORED_FEATURES == N_FEATURES, "features.py and utils.py disagree on the stored frame size"
os.makedirs(ARTIFACTS, exist_ok=True)

# ------------------------------------------------------------------ load data
def _npy_files(action):
    folder = os.path.join(DATA_PATH, action)
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith(".npy"))


# Every class in utils.py (GLOSSES + idle) needs data: the model, actions.npy and the
# live recogniser all use that same list. Report everything that is missing at once.
missing = [a for a in ACTIONS if not _npy_files(a)]
if missing:
    raise SystemExit(
        "No recorded samples for: " + ", ".join(missing) + "\n"
        "  - to record a sign:   python record.py <name>   (and/or record_sequence.py)\n"
        "  - or, to train without it for now, remove it from GLOSSES in utils.py")

X, y = [], []
for idx, action in enumerate(ACTIONS):
    folder = os.path.join(DATA_PATH, action)
    files = _npy_files(action)
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
counts = np.bincount(y, minlength=len(ACTIONS))
too_few = [f"{a} ({c})" for a, c in zip(ACTIONS, counts) if c < 6]
if too_few:
    raise SystemExit("Too few valid samples for: " + ", ".join(too_few) +
                     "\n  Need at least ~6 per class (30+ recommended).")
sign_max = max((c for a, c in zip(ACTIONS, counts) if a != "idle"), default=0)
if "idle" in ACTIONS and counts[ACTIONS.index("idle")] < sign_max:
    print("WARNING: 'idle' has fewer samples than your biggest sign. Record more idle "
          "(varied movement, hands raised but not signing, transitions between signs).")

# ----------------------------------------------------- split (stratified, 70/15/15)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=SEED)
print(f"samples  train {len(X_train)}  val {len(X_val)}  test {len(X_test)}")

class_weight = dict(enumerate(compute_class_weight(
    "balanced", classes=np.arange(len(ACTIONS)), y=y_train)))
print("class weights:", {ACTIONS[k]: round(float(v), 2) for k, v in class_weight.items()})

# Augment the training split only, then build model features.
if AUG_COPIES:
    X_train, y_train = add_augmented(X_train, y_train, AUG_COPIES, seed=SEED)
print(f"train samples after augmentation: {len(X_train)}")

X_train, X_val, X_test = [build_features(a) for a in (X_train, X_val, X_test)]

# Normalisation statistics from the TRAIN split only, one value per feature,
# so they apply to any 30-frame window at inference time. The floor stops
# near-constant features (e.g. a hand that is rarely visible) from exploding.
mu = X_train.mean(axis=(0, 1))
sigma = np.maximum(X_train.std(axis=(0, 1)), 0.05)
X_train, X_val, X_test = [(a - mu) / sigma for a in (X_train, X_val, X_test)]

n_classes = len(ACTIONS)
Y_train = to_categorical(y_train, n_classes)
Y_val = to_categorical(y_val, n_classes)

# ---------------------------------------------------------------------- model
model = Sequential([
    Input(shape=(SEQ_LEN, N_MODEL_FEATURES)),
    GaussianNoise(0.05),                       # light noise, active only in training
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(n_classes, activation="softmax"),
])
model.compile(optimizer=Adam(5e-4), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Normalisation and class order are known before training, so save them first:
# the model file written during training is then always usable with them,
# even if a run is interrupted.
np.save(MU_PATH, mu)
np.save(SIGMA_PATH, sigma)
np.save(ACTIONS_PATH, np.array(ACTIONS))

# ModelCheckpoint writes the model after every epoch in which val_loss improves,
# so the file on disk always holds the best model so far (a crash or Ctrl-C keeps it).
history = model.fit(
    X_train, Y_train, validation_data=(X_val, Y_val),
    epochs=200, batch_size=16, class_weight=class_weight,
    callbacks=[ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
               EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)])

# ----------------------------------------------------------------- best model
# Evaluate and use exactly what is stored on disk (the best epoch).
if not os.path.exists(MODEL_PATH):      # only if val_loss never became a valid number
    model.save(MODEL_PATH)
model = load_model(MODEL_PATH)
val_loss = history.history["val_loss"]
best = int(np.argmin(val_loss))
print(f"Best epoch: {best + 1} of {len(val_loss)} "
      f"(val_loss {val_loss[best]:.4f}, val_accuracy {history.history['val_accuracy'][best]:.2f}) "
      f"- stored in {MODEL_PATH}")
print(f"Model and normalisation are in ./{ARTIFACTS}")

# ------------------------------------------------------------- evaluation
probs = model.predict(X_test, verbose=0)
y_pred = probs.argmax(axis=1)
labels = list(range(n_classes))
print(classification_report(y_test, y_pred, labels=labels,
                            target_names=ACTIONS, zero_division=0))
cm = confusion_matrix(y_test, y_pred, labels=labels)

confused = sorted(((cm[i, j], ACTIONS[i], ACTIONS[j])
                   for i in labels for j in labels if i != j and cm[i, j] > 0), reverse=True)
if confused:
    print("Most confused (true -> predicted):")
    for count, true, pred in confused[:8]:
        print(f"  {true} -> {pred}: {count}")

plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.xticks(labels, ACTIONS, rotation=45)
plt.yticks(labels, ACTIONS)
for i in range(n_classes):
    for j in range(n_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion matrix ({n_classes} classes, held-out test samples)")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS, "confusion_matrix.png"), dpi=120)
print("Saved confusion_matrix.png")