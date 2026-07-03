"""
collect_data.py

Records webcam sequences of hand/body motion (via MediaPipe Holistic) for
training a dynamic sign-language LSTM model.

Usage:
    py -3.11 collect_data.py --level alphabets --action hello --sequences 30

Each run appends new sequences to data/<level>/<action>/ so you can keep
adding more data for the same sign later without overwriting old sequences.

Controls while running:
    SPACE  - start recording the next sequence
    q      - quit early
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import argparse

SEQUENCE_LENGTH = 30      # frames per sequence
DATA_ROOT = "data"


def extract_landmarks(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])


def next_sequence_index(action_path):
    """Finds the next free sequence folder number so re-runs don't overwrite data."""
    if not os.path.exists(action_path):
        return 0
    existing = [int(d) for d in os.listdir(action_path) if d.isdigit()]
    return max(existing) + 1 if existing else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, choices=["alphabets", "basic", "intermediate"])
    parser.add_argument("--action", required=True, help="Name of the sign/word being recorded")
    parser.add_argument("--sequences", type=int, default=30, help="How many new sequences to record this run")
    args = parser.parse_args()

    action_path = os.path.join(DATA_ROOT, args.level, args.action)
    os.makedirs(action_path, exist_ok=True)
    start_idx = next_sequence_index(action_path)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        seq_num = start_idx
        sequences_recorded = 0

        while sequences_recorded < args.sequences:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            # --- Waiting screen before each sequence ---
            waiting = True
            while waiting:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Level: {args.level} | Action: {args.action}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 125), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Sequence {seq_num} ({sequences_recorded}/{args.sequences} done)", (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 125), 2, cv2.LINE_AA)
                cv2.putText(frame, "Press SPACE to record, q to quit", (15, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    waiting = False
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print(f"Stopped early. Recorded {sequences_recorded} new sequences for '{args.action}'.")
                    return

            # --- Record one sequence of SEQUENCE_LENGTH frames ---
            seq_path = os.path.join(action_path, str(seq_num))
            os.makedirs(seq_path, exist_ok=True)

            for frame_num in range(SEQUENCE_LENGTH):
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
                np.save(os.path.join(seq_path, f"{frame_num}.npy"), landmarks)

                cv2.putText(frame, "RECORDING...", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1)

            sequences_recorded += 1
            seq_num += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Recorded {sequences_recorded} new sequences for '{args.action}' at level '{args.level}'.")


if __name__ == "__main__":
    main()
