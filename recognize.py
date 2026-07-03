"""
Real-time FSL recognition across all three curriculum levels.

Loads whichever of model_alphabets.keras / model_basic.keras /
model_intermediate.keras are present (train_sequence.py produces these),
and lets the user switch between levels live with the keyboard.

Usage:
    py -3.11 recognize.py

Controls:
    1 / 2 / 3   switch to Alphabets / Basic / Intermediate
    c           clear the current sentence
    t           speak the current sentence (text-to-speech)
    q           quit
"""

import os
import json
import collections
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from utils import create_hands, extract_landmarks, draw_landmarks

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.85
STABILITY_COUNT = 5  # consecutive matching predictions required before accepting

LEVELS = ("alphabets", "basic", "intermediate")
LEVEL_KEYS = {ord('1'): "alphabets", ord('2'): "basic", ord('3'): "intermediate"}


def speak(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS unavailable: {e}")


def load_level_models():
    """Load every level's model + labels that actually exist on disk."""
    models = {}
    for level in LEVELS:
        model_path = f"model_{level}.keras"
        labels_path = f"signs_{level}.json"
        if os.path.exists(model_path) and os.path.exists(labels_path):
            with open(labels_path) as f:
                signs = json.load(f)
            models[level] = {"model": load_model(model_path), "signs": signs}
            print(f"Loaded {level}: {len(signs)} signs -> {signs}")
        else:
            print(f"Skipping {level}: model_{level}.keras / signs_{level}.json not found.")
    return models


def main():
    models = load_level_models()
    if not models:
        raise RuntimeError(
            "No trained models found. Run train_sequence.py for at least one level first "
            "(e.g. `py -3.11 train_sequence.py --level alphabets`)."
        )

    # Default to the first available level, preferring alphabets if present
    current_level = "alphabets" if "alphabets" in models else next(iter(models))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index 0.")

    hands = create_hands(max_num_hands=2)

    buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    recent_predictions = collections.deque(maxlen=STABILITY_COUNT)
    sentence = []
    last_accepted = None

    print("Controls: 1=Alphabets 2=Basic 3=Intermediate  c=clear  t=speak  q=quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            buffer.append(extract_landmarks(results, max_num_hands=2))
            draw_landmarks(frame, results)

            predicted_label = None
            confidence = 0.0

            if current_level in models and len(buffer) == SEQUENCE_LENGTH:
                level_data = models[current_level]
                input_seq = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)
                preds = level_data["model"].predict(input_seq, verbose=0)[0]
                best_idx = int(np.argmax(preds))
                confidence = float(preds[best_idx])

                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_label = level_data["signs"][best_idx]
                    recent_predictions.append(predicted_label)
                else:
                    recent_predictions.append(None)

                if (
                    len(recent_predictions) == STABILITY_COUNT
                    and len(set(recent_predictions)) == 1
                    and recent_predictions[0] is not None
                    and recent_predictions[0] != last_accepted
                ):
                    sentence.append(recent_predictions[0])
                    last_accepted = recent_predictions[0]
                    buffer.clear()

            available = "/".join(f"{i+1}:{lvl}" for i, lvl in enumerate(LEVELS) if lvl in models)
            cv2.putText(frame, f"Level: {current_level}  [{available}]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            cv2.putText(frame, f"Sign: {predicted_label or '...'} ({confidence:.2f})",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Sentence: " + " ".join(sentence),
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("FSL Recognition", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence = []
                last_accepted = None
            elif key == ord('t'):
                if sentence:
                    speak(" ".join(sentence))
            elif key in LEVEL_KEYS:
                new_level = LEVEL_KEYS[key]
                if new_level in models and new_level != current_level:
                    current_level = new_level
                    buffer.clear()
                    recent_predictions.clear()
                    last_accepted = None
                    print(f"Switched to level: {current_level}")
                elif new_level not in models:
                    print(f"No trained model for '{new_level}' yet.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
