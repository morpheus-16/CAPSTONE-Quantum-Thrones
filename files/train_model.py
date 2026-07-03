"""
train_model.py

Trains an LSTM model on the sequences recorded by collect_data.py, for a
single level (alphabets / basic / intermediate). Saves the model and the
label list so recognize.py can load them later.

Usage:
    py -3.11 train_model.py --level alphabets --epochs 200
"""

import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

SEQUENCE_LENGTH = 30
NUM_FEATURES = 258   # 33*4 (pose) + 21*3 (left hand) + 21*3 (right hand)
DATA_ROOT = "data"
MODEL_ROOT = "models"


def load_dataset(level):
    level_path = os.path.join(DATA_ROOT, level)
    if not os.path.exists(level_path):
        raise FileNotFoundError(f"No data folder found for level '{level}' at {level_path}")

    actions = sorted([d for d in os.listdir(level_path) if os.path.isdir(os.path.join(level_path, d))])
    if not actions:
        raise ValueError(f"No action folders found under {level_path}. Run collect_data.py first.")

    sequences, labels = [], []

    for action_idx, action in enumerate(actions):
        action_path = os.path.join(level_path, action)
        seq_folders = sorted([d for d in os.listdir(action_path) if d.isdigit()], key=int)

        for seq_folder in seq_folders:
            seq_path = os.path.join(action_path, seq_folder)
            frame_files = sorted(
                [f for f in os.listdir(seq_path) if f.endswith(".npy")],
                key=lambda x: int(x.split(".")[0])
            )
            if len(frame_files) != SEQUENCE_LENGTH:
                print(f"Skipping {seq_path}: has {len(frame_files)} frames, expected {SEQUENCE_LENGTH}")
                continue

            window = [np.load(os.path.join(seq_path, f)) for f in frame_files]
            sequences.append(window)
            labels.append(action_idx)

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(actions)).astype(int)
    return X, y, actions


def build_model(num_classes):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        LSTM(64, return_sequences=True, activation='relu'),
        Dropout(0.3),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, choices=["alphabets", "basic", "intermediate"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    print(f"Loading dataset for level '{args.level}'...")
    X, y, actions = load_dataset(args.level)
    print(f"Loaded {X.shape[0]} sequences across {len(actions)} actions: {actions}")

    if X.shape[0] < 10:
        print("Warning: very little data. Consider recording more sequences per action before trusting results.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    model = build_model(num_classes=len(actions))
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc * 100:.2f}% | Test loss: {loss:.4f}")

    os.makedirs(MODEL_ROOT, exist_ok=True)
    model_path = os.path.join(MODEL_ROOT, f"model_{args.level}.keras")
    labels_path = os.path.join(MODEL_ROOT, f"signs_{args.level}.json")

    model.save(model_path)
    with open(labels_path, "w") as f:
        json.dump({"actions": actions}, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved labels to {labels_path}")


if __name__ == "__main__":
    main()
