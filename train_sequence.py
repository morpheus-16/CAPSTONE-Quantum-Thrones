"""
Train an LSTM for one curriculum level (alphabets, basic, or intermediate).

Each level gets its own model, since vocabulary size and sign complexity
differ a lot between "26 alphabet letters" and "intermediate phrases".

Usage:
    py -3.11 train_sequence.py --level alphabets
    py -3.11 train_sequence.py --level basic
    py -3.11 train_sequence.py --level intermediate

Reads data/<level>/<sign>/*.npy and saves:
    model_<level>.keras
    signs_<level>.json   (label order, matches model output indices)
"""

import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

DATA_DIR = "data"
LEVELS = ("alphabets", "basic", "intermediate")


def load_dataset(level):
    level_dir = os.path.join(DATA_DIR, level)
    if not os.path.isdir(level_dir):
        raise RuntimeError(f"No data folder found at '{level_dir}'. Run collect_sequence.py first.")

    signs = sorted(d for d in os.listdir(level_dir) if os.path.isdir(os.path.join(level_dir, d)))
    if not signs:
        raise RuntimeError(f"No sign folders found under '{level_dir}'. Run collect_sequence.py first.")

    X, y = [], []
    for label_index, sign in enumerate(signs):
        sign_dir = os.path.join(level_dir, sign)
        files = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
        if not files:
            print(f"Warning: no sequences found for sign '{sign}', skipping.")
            continue
        for fname in files:
            X.append(np.load(os.path.join(sign_dir, fname)))
            y.append(label_index)

    if not X:
        raise RuntimeError(f"No sequences loaded for level '{level}'.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), signs


def build_model(seq_len, num_features, num_classes):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(seq_len, num_features)),
        LSTM(128, return_sequences=True, activation="tanh"),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation="tanh"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train an FSL sign LSTM for one level.")
    parser.add_argument("--level", required=True, choices=LEVELS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    X, y, signs = load_dataset(args.level)
    print(f"[{args.level}] Loaded {len(X)} sequences across {len(signs)} signs: {signs}")

    num_classes = len(signs)
    y_cat = to_categorical(y, num_classes=num_classes)

    min_per_class = np.min(np.bincount(y))
    stratify = y if min_per_class >= 2 else None
    if stratify is None:
        print("Warning: at least one sign has fewer than 2 sequences; skipping stratified split.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=stratify
    )

    model = build_model(seq_len=X.shape[1], num_features=X.shape[2], num_classes=num_classes)
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"[{args.level}] Validation accuracy: {val_acc:.3f}")

    model_path = f"model_{args.level}.keras"
    labels_path = f"signs_{args.level}.json"

    model.save(model_path)
    with open(labels_path, "w") as f:
        json.dump(signs, f, indent=2)

    print(f"Saved model to {model_path} and labels to {labels_path}")


if __name__ == "__main__":
    main()
