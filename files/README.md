# FSL Recognition Pipeline — Fresh Start

A from-scratch pipeline for collecting your own sign language data, training
per-level LSTM models, and running real-time recognition. Built around
MediaPipe Holistic (pose + both hands) feeding a Keras LSTM.

## Requirements

- Python 3.11 (`py -3.11 ...`) — required for MediaPipe/TensorFlow compatibility
- Packages: `opencv-python`, `mediapipe`, `tensorflow`, `scikit-learn`, `numpy`

```
py -3.11 -m pip install opencv-python mediapipe tensorflow scikit-learn numpy
```

## Folder Structure

Run all three scripts from the same project root. They create/expect:

```
project_root/
├── data/
│   ├── alphabets/<action_name>/<sequence_num>/<frame_num>.npy
│   ├── basic/<action_name>/<sequence_num>/<frame_num>.npy
│   └── intermediate/<action_name>/<sequence_num>/<frame_num>.npy
├── models/
│   ├── model_alphabets.keras   signs_alphabets.json
│   ├── model_basic.keras       signs_basic.json
│   └── model_intermediate.keras  signs_intermediate.json
├── collect_data.py
├── train_model.py
└── recognize.py
```

Each sequence is 30 frames; each frame is a 258-value landmark vector
(33 pose landmarks × 4 + 21 left-hand × 3 + 21 right-hand × 3).

## Step 1 — Collect Data

Record webcam sequences for one sign at a time:

```
py -3.11 collect_data.py --level alphabets --action letter_j --sequences 30
```

- `--level`: `alphabets`, `basic`, or `intermediate`
- `--action`: name of the sign (this becomes the folder name and label)
- `--sequences`: how many new sequences to record this run (default 30)

Controls while running:
- `SPACE` — start recording the next sequence
- `q` — stop early

Re-running the same `--level --action` combo **adds** more sequences instead
of overwriting existing ones, so you can top up data for a sign later.

**Tip:** more sequences = a more reliable model. 30 is a reasonable minimum
for a dynamic sign; 80+ (like your earlier prototype used) trains more
robustly, especially if you vary lighting, distance from camera, and speed
of signing across sequences.

## Step 2 — Train a Model

Once a level has data for its actions:

```
py -3.11 train_model.py --level alphabets --epochs 200
```

This scans `data/alphabets/` for action subfolders, loads all valid
sequences, splits train/test, trains an LSTM, and saves:

- `models/model_alphabets.keras`
- `models/signs_alphabets.json` (the ordered label list)

Repeat for `basic` and `intermediate` once you've collected data for those
levels. Training stops early if validation loss stops improving
(`EarlyStopping`, patience 20).

## Step 3 — Run Recognition

```
py -3.11 recognize.py
```

- Loads whichever level models exist in `models/` at startup (skips missing
  ones with a warning — you don't need all three levels trained to run this)
- `1` / `2` / `3` — switch between Alphabets / Basic / Intermediate live
- `c` — clear the current prediction buffer
- `q` — quit
- Predictions below 60% confidence show as "Uncertain..." instead of a
  guess — adjust `CONFIDENCE_THRESHOLD` in `recognize.py` if it's too
  strict/loose for your data.

## Adding More Signs Later

Just run `collect_data.py` again with a new `--action`, then re-run
`train_model.py --level <level>` — it automatically picks up every action
folder present, so the model retrains on the full updated set each time.

## Notes

- All three scripts share the same landmark extraction and sequence length,
  so data collected today will still work if you retrain later.
- If you ever change `SEQUENCE_LENGTH` in `collect_data.py`, you must also
  update it in `train_model.py` and `recognize.py`, and re-record data —
  old sequences won't match the new frame count.
