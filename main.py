import os
import cv2
import numpy as np
import json
import math
import time
import threading
import speech_recognition as sr

from google import genai
from google.genai import types

from hand_tracking import HandTracker
from voice_assistant import VoiceAssistant

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_ID = "gemini-3-flash-preview"

CAM_W, CAM_H = 640, 480

# --- speed knobs ---
ORB_FEATURES = 1500      # was 3000
ORB_SCALE = 0.5          # run ORB on a half-size image
ORB_EVERY_N_FRAMES = 3   # reuse the last homography in between
MAX_HOMOGRAPHY_AGE = 12  # frames before a cached homography is discarded
MIN_GOOD_MATCHES = 25

# --- accuracy knobs ---
MIN_HAND_SCORE = 0.8
FINGER_SMOOTHING = 0.5
MIN_THRESHOLD_PX = 22
MAX_THRESHOLD_PX = 90

# --- audio pacing ---
DIRECTION_COOLDOWN = 1.4        # min gap between any two direction cues
SAME_DIRECTION_COOLDOWN = 2.6   # min gap before repeating the same cue
CONFIRM_NUDGE_SEC = 20.0
PANEL_LOST_WARN_SEC = 3.0

CONFIRM_KEYWORDS = {
    # intended words
    "done", "yes", "pressed", "next", "ready", "ok", "okay", "yep", "yeah",
    "finished", "complete", "completed", "did", "got",
    # common mishearings of "done" / "next" / "ok"
    "dawn", "don", "down", "dun", "none", "gone", "one", "won", "sun", "fun",
    "text", "nest", "necks", "nicks", "oak", "okey", "k",
}

if not API_KEY:
    raise SystemExit(
        "GEMINI_API_KEY is not set.\n"
        "PowerShell (permanent):  setx GEMINI_API_KEY \"your_key\"  then reopen VS Code\n"
        "PowerShell (this shell): $env:GEMINI_API_KEY=\"your_key\""
    )

client = genai.Client(api_key=API_KEY)


def get_appliance_steps(image_bytes, user_goal):
    prompt_text = f"""
    You are an Assistive Navigation Agent for visually impaired users.
    User Goal: "{user_goal}"
    Target Language: "English"
    Examine the appliance control panel image:
    1. Identify the appliance type.
    2. Determine the EXACT sequential order of buttons/controls that MUST be
       pressed to achieve the user's goal.

    CRITICAL RULES:
    - You can group multiple presses of the SAME button into a single step
      (e.g., "Press the 10s button 3 times").
    - box_2d scale: 0-1000, in the order [ymin, xmin, ymax, xmax], and it must
      tightly bound the specific control, not the whole panel.
    - Keep each "text" under 15 words. It will be read aloud.
    - If the panel is not completely visible in the image sent (for example if only 1/3rd of the panel is in the camera frame), inform the user to take a better photo by moving left, right, up or down as required. DO NOT return the steps in this case simple tell the user recapture and try again. Do not return ANY STEPS, simply return empty list.
    Return PURE JSON in this exact structure:
    {{
        "device_name": "Appliance Name",
        "steps": [
            {{
                "order": 1,
                "button_name": "Timer Button",
                "text": "Press the 30 second timer button three times",
                "action_type": "tap",
                "box_2d": [100, 200, 150, 250]
            }}
        ]
    }}
    """
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt_text
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = json.loads(response.text)
        return data[0] if isinstance(data, list) else data
    except Exception as e:
        print(f"API Error: {e}")
        return None


def build_steps(api_data, ref_shape):
    """Convert Gemini's 0-1000 boxes into reference-image pixels."""
    img_h, img_w = ref_shape[:2]
    out = []

    for step in api_data.get("steps", []):
        box = step.get("box_2d")
        if not box or len(box) != 4:
            continue

        ymin, xmin, ymax, xmax = [max(0, min(1000, float(v))) for v in box]
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        if xmax < xmin:
            xmin, xmax = xmax, xmin

        center_x = int(((xmin + xmax) / 2000.0) * img_w)
        center_y = int(((ymin + ymax) / 2000.0) * img_h)
        box_w = ((xmax - xmin) / 1000.0) * img_w
        box_h = ((ymax - ymin) / 1000.0) * img_h

        # Adaptive hit radius: roughly half the smaller side of the button.
        base_radius = max(MIN_THRESHOLD_PX, 0.5 * min(box_w, box_h))

        out.append({
            "name": step.get("button_name", "Button"),
            "instruction": step.get("text", "Press the button"),
            "ref_coord": (center_x, center_y),
            "base_radius": base_radius
        })

    return out


def describe_direction(dx, dy, thr):
    """dx, dy = target - fingertip, in pixels."""
    parts = []
    dead = thr * 0.8

    if abs(dx) > dead:
        parts.append("right" if dx > 0 else "left")
    if abs(dy) > dead:
        parts.append("down" if dy > 0 else "up")

    if not parts:
        return "Almost there. Hold steady."

    dist = math.hypot(dx, dy)
    qualifier = "slightly " if dist < thr * 2.5 else ""
    return "Move " + qualifier + " and ".join(parts)


# ==========================================
# MICROPHONE SETUP
# ==========================================
recognizer = sr.Recognizer()
recognizer.pause_threshold = 2.0
recognizer.dynamic_energy_threshold = True

mic = sr.Microphone()

mic_lock = threading.Lock()
user_said_done = False
last_keyword_time = 0.0
last_heard_text = ""


def mic_callback(_recognizer, audio):
    """Runs in a background thread whenever sound is detected."""
    global user_said_done, last_keyword_time, last_heard_text
    try:
        text = _recognizer.recognize_google(audio).lower()
    except (sr.UnknownValueError, sr.RequestError):
        return

    print(f"[Mic heard]: {text}")

    # Token matching, so "I don't know" can't accidentally contain "no"
    tokens = {t.strip(".,!?") for t in text.split()}

    with mic_lock:
        last_heard_text = text
        if tokens & CONFIRM_KEYWORDS:
            user_said_done = True
            last_keyword_time = time.time()
            print("[Mic] >>> CONFIRMATION DETECTED")


print("Calibrating microphone for ambient noise... Please wait.")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=2.0)
print("Microphone ready.")

# ==========================================
# INITIALIZATION
# ==========================================
tracker = HandTracker(
    detection_confidence=0.8,
    tracking_confidence=0.8,
    smoothing=FINGER_SMOOTHING
)

voice = VoiceAssistant(
    rate=185,
    repeat_interval=1.2,
    voice_index=2,      # Microsoft Zira (falls back to 0 if unavailable)
    volume=1.0
)

orb = cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2, nlevels=8)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

reference_img = None
kp_ref, des_ref = None, None
steps_data = []
current_step = 0

stop_listening = None

# ==========================================
# PHASE 1: CAPTURE & AUDIO GOAL PROCESSING
# ==========================================
setup_ok = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, "Align panel and press 'c' to capture", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Setup Phase", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('c'):
        reference_img = frame.copy()
        reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
        ref_small = cv2.resize(reference_gray, None, fx=ORB_SCALE, fy=ORB_SCALE)
        kp_ref, des_ref = orb.detectAndCompute(ref_small, None)

        if des_ref is None or len(kp_ref) < 50:
            print("Too few features in that shot. Improve lighting and try again.")
            voice.say_and_wait("That image is too blurry. Please try again.")
            continue

        print(f"[ORB] {len(kp_ref)} reference keypoints.")

        ok, encoded_img = cv2.imencode('.jpg', reference_img)
        raw_bytes = encoded_img.tobytes()

        busy = frame.copy()
        cv2.putText(busy, "Listening for your goal...", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Setup Phase", busy)
        cv2.waitKey(1)

        voice.say_and_wait("Panel captured. What would you like to do?")

        goal = ""
        attempts = 0
        while not goal and attempts < 4:
            attempts += 1
            try:
                print("Listening for goal...")
                with mic as source:
                    audio = recognizer.listen(source, timeout=8, phrase_time_limit=15)
                goal = recognizer.recognize_google(audio)
                print(f"Goal heard: {goal}")
                voice.say_and_wait(f"Understood. Getting instructions to {goal}.")
            except sr.WaitTimeoutError:
                voice.say_and_wait("I didn't hear anything. What would you like to do?")
            except sr.UnknownValueError:
                voice.say_and_wait("I didn't catch that. Please say it again.")
            except sr.RequestError:
                print("Network error connecting to speech recognition.")
                break

        if not goal:
            print("No goal captured. Press 'c' to retry.")
            continue

        print("Sending to Gemini API...")
        busy = frame.copy()
        cv2.putText(busy, "Thinking...", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow("Setup Phase", busy)
        cv2.waitKey(1)

        api_data = get_appliance_steps(raw_bytes, goal)

        if api_data:
            steps_data = build_steps(api_data, reference_img.shape)

        if steps_data:
            print(f"Device: {api_data.get('device_name', 'Unknown')}")
            print(f"Successfully mapped {len(steps_data)} steps.")
            cv2.destroyWindow("Setup Phase")
            stop_listening = recognizer.listen_in_background(mic, mic_callback)
            setup_ok = True
            break
        else:
            print("Failed to get valid steps. Try capturing again.")
            voice.say_and_wait("I could not read that panel. Let's try again.")

# ==========================================
# PHASE 2: LIVE NAVIGATION LOOP
# ==========================================
cached_H = None
homography_age = 999
frame_idx = 0

last_direction = ""
last_direction_time = 0.0
awaiting_confirmation = False
last_prompt_time = 0.0
panel_lost_since = None
panel_warned = False

fps_t0 = time.time()
fps_count = 0
fps = 0.0

if setup_ok and steps_data:
    print("Starting Live Navigation...")
    voice.speak(f"Starting step 1. {steps_data[0]['instruction']}")

while setup_ok:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    now = time.time()

    fps_count += 1
    if now - fps_t0 >= 1.0:
        fps = fps_count / (now - fps_t0)
        fps_count = 0
        fps_t0 = now

    # ---------- finished? ----------
    if current_step >= len(steps_data):
        cv2.putText(frame, "Operation Complete", (90, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("Navigation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    target_data = steps_data[current_step]
    ref_pt_small = np.float32(
        [[(target_data["ref_coord"][0] * ORB_SCALE,
           target_data["ref_coord"][1] * ORB_SCALE)]]
    ).reshape(-1, 1, 2)

    # ---------- ORB tracking (every Nth frame) ----------
    if frame_idx % ORB_EVERY_N_FRAMES == 0 or cached_H is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=ORB_SCALE, fy=ORB_SCALE)
        kp_frame, des_frame = orb.detectAndCompute(small, None)

        new_H = None
        if des_frame is not None and des_ref is not None and len(kp_frame) > 10:
            matches = bf.knnMatch(des_ref, des_frame, k=2)
            good = [m for pair in matches if len(pair) == 2
                    for m, n in [pair] if m.distance < 0.70 * n.distance]

            if len(good) >= MIN_GOOD_MATCHES:
                src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if H is not None and mask is not None and int(mask.sum()) >= 15:
                    new_H = H

        if new_H is not None:
            cached_H = new_H
            homography_age = 0
        else:
            homography_age += ORB_EVERY_N_FRAMES
    else:
        homography_age += 1

    if homography_age > MAX_HOMOGRAPHY_AGE:
        cached_H = None

    live_target_coord = None
    hit_radius = target_data["base_radius"]

    if cached_H is not None:
        live_pt = cv2.perspectiveTransform(ref_pt_small, cached_H)
        live_target_coord = (int(live_pt[0][0][0] / ORB_SCALE),
                             int(live_pt[0][0][1] / ORB_SCALE))

        # Scale the hit radius with how zoomed-in the panel currently is
        try:
            zoom = math.sqrt(abs(np.linalg.det(cached_H[:2, :2])))
        except Exception:
            zoom = 1.0
        hit_radius = float(np.clip(target_data["base_radius"] * zoom,
                                   MIN_THRESHOLD_PX, MAX_THRESHOLD_PX))

    # ---------- panel-lost audio warning ----------
    if live_target_coord is None:
        if panel_lost_since is None:
            panel_lost_since = now
        elif not panel_warned and (now - panel_lost_since) > PANEL_LOST_WARN_SEC \
                and not awaiting_confirmation:
            voice.speak("I lost sight of the panel. Please move the camera back.")
            panel_warned = True
    else:
        panel_lost_since = None
        panel_warned = False

    # ---------- hand ----------
    results = tracker.process(frame)
    hand_landmarks, hand_score = tracker.best_hand(results, min_score=MIN_HAND_SCORE)

    finger = None
    if hand_landmarks is not None:
        tracker.draw_landmarks(frame, hand_landmarks)
        finger = tracker.get_fingertip(frame, hand_landmarks, smooth=True)
        cv2.circle(frame, finger, 12, (0, 255, 0), -1)

    # ---------- confirmation (works even if hand / panel is not visible) ----------
    with mic_lock:
        heard_done = user_said_done
        flash = (now - last_keyword_time) < 0.8

    if awaiting_confirmation:
        if heard_done:
            with mic_lock:
                user_said_done = False

            current_step += 1
            awaiting_confirmation = False
            last_direction = ""
            tracker.reset_smoothing()

            if current_step < len(steps_data):
                voice.speak("Step complete. Next step. "
                            + steps_data[current_step]["instruction"])
            else:
                voice.speak("All steps completed successfully.")

        elif now - last_prompt_time > CONFIRM_NUDGE_SEC:
            voice.speak("Please say done when you have finished the step.")
            last_prompt_time = now

    # ---------- guidance ----------
    elif live_target_coord is not None and finger is not None:
        tx, ty = live_target_coord
        dx = tx - finger[0]
        dy = ty - finger[1]
        distance = math.hypot(dx, dy)

        cv2.line(frame, finger, live_target_coord, (255, 255, 0), 2)

        if distance < hit_radius:
            voice.speak("Button reached. Press it, then say done.")
            awaiting_confirmation = True
            last_prompt_time = now
            with mic_lock:
                user_said_done = False
        else:
            direction = describe_direction(dx, dy, hit_radius)
            cooldown = SAME_DIRECTION_COOLDOWN if direction == last_direction \
                else DIRECTION_COOLDOWN

            if (now - last_direction_time) >= cooldown and not voice.is_speaking:
                voice.speak(direction)
                last_direction = direction
                last_direction_time = now

            cv2.putText(frame, direction, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)

    # ---------- HUD ----------
    if live_target_coord is not None:
        tx, ty = live_target_coord
        cv2.circle(frame, live_target_coord, int(hit_radius), (0, 140, 255), 2)
        cv2.circle(frame, live_target_coord, 14, (0, 0, 255), -1)
        cv2.putText(frame, target_data["name"], (tx - 25, ty - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Searching for panel...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    if awaiting_confirmation:
        status = "Awaiting verbal confirmation..."
        status_color = (0, 165, 255)
    elif voice.is_speaking:
        status = "Speaking..."
        status_color = (0, 255, 255)
    else:
        status = "Tracking..."
        status_color = (0, 255, 0)

    cv2.putText(frame, status, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"Step: {current_step + 1}/{len(steps_data)}", (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:4.1f}  hand:{hand_score:.2f}  r:{int(hit_radius)}",
                (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, "q quit | n next | b back | r repeat", (30, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Green border flash the instant a keyword is heard
    if flash:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 14)
        cv2.putText(frame, "HEARD", (w - 130, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Navigation", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):          # manual skip, useful during demos
        with mic_lock:
            user_said_done = True
        if not awaiting_confirmation:
            awaiting_confirmation = True
            last_prompt_time = now
    elif key == ord('b') and current_step > 0:
        current_step -= 1
        awaiting_confirmation = False
        last_direction = ""
        voice.speak("Going back. " + steps_data[current_step]["instruction"])
    elif key == ord('r'):
        voice.speak(steps_data[current_step]["instruction"])

# ==========================================
# CLEANUP
# ==========================================
if stop_listening:
    stop_listening(wait_for_stop=False)
voice.stop()
tracker.close()
cap.release()
cv2.destroyAllWindows()
print("Shut down cleanly.")