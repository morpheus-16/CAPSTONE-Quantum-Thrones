# FSL Recognition (MediaPipe + OpenCV + LSTM)

Motion-based Filipino Sign Language recognition, covering three curriculum
levels: **Alphabets**, **Basic**, and **Intermediate**. Every sign — including
alphabet letters that involve movement — is captured and recognized as a
short motion sequence, not a single static image.

## Why sequences, not single images

A CNN on a single frame can't tell "hand held still" from "hand mid-motion."
This system tracks 21 hand landmarks per hand (up to 2 hands) with
MediaPipe every frame, buffers 30 frames into a sequence, and feeds that
sequence into an LSTM. The LSTM learns the *trajectory*, which is what
actually distinguishes many FSL signs.

## Requirements

- Python 3.11 (MediaPipe does not support 3.14; use `py -3.11` on Windows
  if that's your default interpreter, matching your existing project setup).
- A webcam.

```bash
py -3.11 -m pip install -r requirements.txt
```

## Project layout

```
fsl_recognition/
  utils.py              # MediaPipe landmark extraction, shared by every script
  collect_sequence.py   # Record motion sequences for one sign at a time
  train_sequence.py     # Train an LSTM for one level
  recognize.py          # Real-time recognition across all trained levels
  data/
    alphabets/<sign>/sequence_*.npy
    basic/<sign>/sequence_*.npy
    intermediate/<sign>/sequence_*.npy
  model_alphabets.keras     # produced by training, not committed until trained
  signs_alphabets.json
  model_basic.keras
  signs_basic.json
  model_intermediate.keras
  signs_intermediate.json
```

Training and inference are fully separate: `collect_sequence.py` and
`train_sequence.py` are developer tools you run offline. `recognize.py`
only ever *loads* the finished `.keras` + `.json` files — it never trains
anything itself.

## 1. Collect data

Run once per sign, per level. Each run records `--sequences` motion clips
(default 30, matching your earlier prototype's dataset size).

```bash
py -3.11 collect_sequence.py --level alphabets --sign letter_a --sequences 30
py -3.11 collect_sequence.py --level alphabets --sign letter_j --sequences 30
py -3.11 collect_sequence.py --level basic --sign thank_you --sequences 30
py -3.11 collect_sequence.py --level intermediate --sign how_are_you --sequences 30
```

Press `s` to record each sequence, `q` to stop early. Re-running with the
same `--level`/`--sign` appends more sequences instead of overwriting.

**Tips for a dataset that generalizes:**
- Record more than one signer if possible.
- Vary lighting, background, and camera distance across sessions.
- Keep a simple log (spreadsheet or markdown) of which folder name maps to
  which actual sign — useful for your defense methodology section.

## 2. Train

One model per level, trained independently:

```bash
py -3.11 train_sequence.py --level alphabets
py -3.11 train_sequence.py --level basic
py -3.11 train_sequence.py --level intermediate
```

Each run produces `model_<level>.keras` and `signs_<level>.json` in the
project folder. You only need to train the levels you've collected data for.

## 3. Run recognition

```bash
py -3.11 recognize.py
```

Loads whichever level models are present. Controls shown on screen:

| Key | Action |
|-----|--------|
| 1 / 2 / 3 | Switch to Alphabets / Basic / Intermediate |
| c | Clear the current sentence |
| t | Speak the current sentence (text-to-speech) |
| q | Quit |

## Tuning

In `recognize.py`:
- `CONFIDENCE_THRESHOLD` (default 0.85) — raise if you're getting false
  positives, lower if correct signs aren't being accepted.
- `STABILITY_COUNT` (default 5) — how many consecutive matching predictions
  are required before a sign is accepted into the sentence. Higher = more
  stable but slower to respond.

## Re-training later

Adding a new sign or more data doesn't require touching any code — just
run `collect_sequence.py` for the new/additional data, then re-run
`train_sequence.py --level <level>` to regenerate that level's model file.
`recognize.py` will pick up the new model automatically on next launch.
