"""
Record hand-landmark motion sequences for any sign, in any level.

All three levels (alphabets, basic, intermediate) are treated the same way
here: every sign is captured as a short motion sequence, since FSL signs
--including several alphabet letters-- can involve movement, not just a
held handshape.

Usage:
    py -3.11 collect_sequence.py --level alphabets --sign letter_j --sequences 30
    py -3.11 collect_sequence.py --level basic --sign thank_you --sequences 30
    py -3.11 collect_sequence.py --level intermediate --sign how_are_you --sequences 30

Saves to: data/<level>/<sign>/sequence_N.npy
Re-running with the same --level/--sign appends more sequences instead of
overwriting existing ones.
"""

import os
import argparse
import cv2
import numpy as np

from utils import create_hands, extract_landmarks, draw_landmarks

SEQUENCE_LENGTH = 30
DATA_DIR = "data"
LEVELS = ("alphabets", "basic", "intermediate")


def main():
    parser = argparse.ArgumentParser(description="Collect FSL sign motion sequences.")
    parser.add_argument("--level", required=True, choices=LEVELS, help="Which curriculum level this sign belongs to.")
    parser.add_argument("--sign", required=True, help="Name of the sign, e.g. 'letter_a' or 'thank_you'.")
    parser.add_argument("--sequences", type=int, default=30, help="Sequences to record this run.")
    parser.add_argument("--seq_len", type=int, default=SEQUENCE_LENGTH, help="Frames per sequence.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--max_hands", type=int, default=2, help="Max hands to track (1 or 2).")
    args = parser.parse_args()

    sign_dir = os.path.join(DATA_DIR, args.level, args.sign)
    os.makedirs(sign_dir, exist_ok=True)

    existing = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
    start_index = len(existing)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    hands = create_hands(max_num_hands=args.max_hands)

    print(f"[{args.level}] Collecting {args.sequences} sequences for sign '{args.sign}', "
          f"starting at index {start_index}.")
    print("Press 's' to record a sequence, 'q' to stop early.")

    collected = 0
    try:
        while collected < args.sequences:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()
            cv2.putText(display, f"[{args.level}] {args.sign}  "
                                  f"{start_index + collected + 1}/{start_index + args.sequences}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Press 's' to record, 'q' to quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Collect FSL Data", display)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                sequence = []
                for frame_num in range(args.seq_len):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    sequence.append(extract_landmarks(results, max_num_hands=args.max_hands))

                    draw_landmarks(frame, results)
                    cv2.putText(frame, f"Recording {frame_num + 1}/{args.seq_len}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Collect FSL Data", frame)
                    cv2.waitKey(1)

                if len(sequence) == args.seq_len:
                    save_path = os.path.join(sign_dir, f"sequence_{start_index + collected}.npy")
                    np.save(save_path, np.array(sequence, dtype=np.float32))
                    print(f"Saved {save_path}")
                    collected += 1
                else:
                    print("Sequence cut short, discarded. Try again.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    print(f"Done. {collected} new sequences collected for [{args.level}] '{args.sign}'.")


if __name__ == "__main__":
    main()
