# ANVI — Assistive Navigation for Visually Impaired users

ANVI helps blind and low-vision people operate appliances with touch control panels (microwaves, washing machines, ovens, etc.). Point a webcam at the panel, say what you want to do, and ANVI guides your finger to each button with spoken directions.

## How it works

1. **Capture** — Align the camera with the control panel and press `c`.
2. **Speak your goal** — e.g. *"Heat food for 30 seconds."* Speech is transcribed with Google Speech Recognition.
3. **Plan** — The panel image and goal are sent to **Google Gemini**, which identifies the appliance and returns the ordered list of buttons to press, with bounding boxes.
4. **Guide** — During live navigation:
   - **ORB feature matching + homography** keeps the target button located as the camera moves.
   - **MediaPipe Hands** tracks your index fingertip.
   - ANVI speaks directions ("Move slightly left and up") until your finger reaches the button.
5. **Confirm** — Press the button and say **"done"** (or "next", "ok", "yes") to move to the next step.

## Project structure

| File | Purpose |
|------|---------|
| `main.py` | Main app: capture, voice goal, Gemini planning, live navigation loop |
| `hand_tracking.py` | `HandTracker` — MediaPipe wrapper with fingertip smoothing and a confidence gate |
| `voice_assistant.py` | `VoiceAssistant` — threaded, non-blocking text-to-speech (SAPI on Windows, pyttsx3 elsewhere) |
| `list_voices.py` | Lists installed TTS voices (to pick a `voice_index`) |
| `test_voice.py` | Quick TTS sanity check |

## Requirements

- Python 3.10
- A webcam and a microphone
- Internet connection (Gemini API + Google Speech Recognition)
- A [Gemini API key](https://aistudio.google.com/apikey)
- Windows recommended (uses SAPI voices via `pywin32`); pyttsx3 is used as a fallback on other platforms

## Setup

```bash
git clone https://github.com/nityasravanthi/anvi.git
cd anvi
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Set your Gemini API key:

```powershell
# PowerShell (current session)
$env:GEMINI_API_KEY = "your_key"

# PowerShell (permanent — reopen the terminal afterwards)
setx GEMINI_API_KEY "your_key"
```

```bash
# macOS / Linux
export GEMINI_API_KEY="your_key"
```

## Usage

```bash
python main.py
```

### Keyboard controls

| Key | Phase | Action |
|-----|-------|--------|
| `c` | Setup | Capture the control panel |
| `q` | Any | Quit |
| `n` | Navigation | Skip to the next step |
| `b` | Navigation | Go back one step |
| `r` | Navigation | Repeat the current instruction |

### Voice commands

- During setup: say your goal after the prompt *"What would you like to do?"*
- During navigation: say **"done"**, **"next"**, **"ok"** or **"yes"** after pressing a button.

## Configuration

Tuning knobs are at the top of `main.py`:

- `MODEL_ID` — Gemini model
- `ORB_FEATURES`, `ORB_SCALE`, `ORB_EVERY_N_FRAMES` — tracking speed vs. accuracy
- `MIN_HAND_SCORE`, `FINGER_SMOOTHING` — hand-tracking stability
- `DIRECTION_COOLDOWN`, `SAME_DIRECTION_COOLDOWN` — how often directions are spoken

Run `python list_voices.py` and change `voice_index` in `main.py` to pick a different voice.

## Tips

- Use good, even lighting — panel tracking relies on visual features.
- Capture the **entire** panel in the frame; ANVI will ask you to retake the photo otherwise.
- Keep only one hand in view during navigation.
