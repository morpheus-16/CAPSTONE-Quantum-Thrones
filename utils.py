"""
Shared MediaPipe Hands helpers used by collect_data.py, train_model.py, and recognize.py.

Landmark encoding:
- Up to 2 hands tracked.
- Each hand contributes 21 landmarks * (x, y, z) = 63 values.
- Hands are ordered left-to-right by wrist x-position so the same physical
  hand consistently lands in the same slice of the feature vector.
- A missing hand is zero-padded rather than omitted, so every frame always
  produces a fixed-length vector (126 values for 2 hands).
"""

import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

LANDMARKS_PER_HAND = 21
VALUES_PER_HAND = LANDMARKS_PER_HAND * 3  # x, y, z


def create_hands(static_image_mode=False, max_num_hands=2,
                  min_detection_confidence=0.6, min_tracking_confidence=0.6):
    """Create a MediaPipe Hands processor with sane defaults for live video."""
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_landmarks(results, max_num_hands=2):
    """Flatten MediaPipe hand results into a fixed-length numpy array."""
    flat = np.zeros(VALUES_PER_HAND * max_num_hands, dtype=np.float32)

    if not results.multi_hand_landmarks:
        return flat

    hands_data = []
    for hand_landmarks in results.multi_hand_landmarks:
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        wrist_x = hand_landmarks.landmark[0].x
        hands_data.append((wrist_x, coords))

    # Sort left-to-right so the same hand tends to occupy the same feature slice
    hands_data.sort(key=lambda h: h[0])

    for i, (_, coords) in enumerate(hands_data[:max_num_hands]):
        start = i * VALUES_PER_HAND
        flat[start:start + VALUES_PER_HAND] = coords

    return flat


def draw_landmarks(image, results):
    """Draw hand skeletons onto a BGR frame in place, for visual feedback."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 120, 255), thickness=2),
            )
