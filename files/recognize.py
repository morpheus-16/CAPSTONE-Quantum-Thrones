"""
recognize.py

Real-time sign recognition using the models trained by train_model.py.
Loads all available level models (alphabets / basic / intermediate) at
startup and lets you switch between them live.

Usage:
    py -3.11 recognize.py

Controls:
    1 = Alphabets   2 = Basic   3 = Intermediate
    c = clear sentence buffer
    q = quit
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from collections import deque
from tensorflow import keras

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 60.0
MODEL_ROOT = "models"

LEVELS = {
    ord('1'): "alphabets",
    ord('2'): "basic",
    ord('3'): "intermediate",
}


def extract_landmarks(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])


def load_level_models():
    """Loads whichever level models are present; skips missing ones with a warning."""
    loaded = {}
    for level in ["alphabets", "basic", "intermediate"]:
        model_path = os.path.join(MODEL_ROOT, f"model_{level}.keras")
        labels_path = os.path.join(MODEL_ROOT, f"signs_{level}.json")

        if not (os.path.exists(model_path) and os.path.exists(labels_path)):
            print(f"Skipping {level}: {model_path} / {labels_path} not found.")
            continue

        model = keras.models.load_model(model_path)
        with open(labels_path, "r") as f:
            actions = json.load(f)["actions"]

        loaded[level] = {"model": model, "actions": np.array(actions)}
        print(f"Loaded {level}: {len(actions)} signs -> {actions}")

    return loaded


def main():
    models = load_level_models()
    if not models:
        print("No trained models found in ./models. Run train_model.py first.")
        return

    current_level = next(iter(models))  # default to first available level
    frame_sequence = deque(maxlen=SEQUENCE_LENGTH)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    current_sign = "Waiting..."
    accuracy = 0.0

    print("Controls: 1=Alphabets 2=Basic 3=Intermediate  c=clear  q=quit")
    print(f"Current level: {current_level}")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            frame.flags.writeable = True

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            landmarks = extract_landmarks(results)
            frame_sequence.append(landmarks)

            if len(frame_sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(list(frame_sequence), axis=0)
                model_data = models[current_level]
                probabilities = model_data["model"].predict(input_data, verbose=0)[0]
                max_idx = np.argmax(probabilities)
                predicted_word = model_data["actions"][max_idx]
                predicted_acc = probabilities[max_idx] * 100

                if predicted_acc >= CONFIDENCE_THRESHOLD:
                    current_sign, accuracy = predicted_word, predicted_acc
                else:
                    current_sign, accuracy = "Uncertain...", predicted_acc

            cv2.rectangle(frame, (0, 0), (640, 70), (45, 45, 45), -1)
            cv2.putText(frame, f"Level: {current_level.upper()}", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Sign: {str(current_sign).upper()} | Accuracy: {accuracy:.1f}%", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 125), 2, cv2.LINE_AA)

            cv2.imshow('FSL Real-Time Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                frame_sequence.clear()
                current_sign, accuracy = "Waiting...", 0.0
            elif key in LEVELS:
                requested_level = LEVELS[key]
                if requested_level in models:
                    current_level = requested_level
                    frame_sequence.clear()
                    current_sign, accuracy = "Waiting...", 0.0
                    print(f"Switched to level: {current_level}")
                else:
                    print(f"Level '{requested_level}' has no trained model yet.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
