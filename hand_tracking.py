import cv2
import mediapipe as mp


class HandTracker:
    """
    MediaPipe hand tracker with fingertip smoothing and a confidence gate.

    Note on confidence: MediaPipe Hands does not expose a per-landmark
    detection score. The value in `multi_handedness` is the left/right
    classification score, which is a reasonable proxy for "is this really
    a hand" -- a half-occluded or glitchy hand usually drops below 0.8.
    The real detection gate is min_detection_confidence below.
    """

    def __init__(
        self,
        max_num_hands=1,
        detection_confidence=0.8,
        tracking_confidence=0.8,
        smoothing=0.5,          # 0 = no smoothing, 0.8 = very heavy
        reset_after_misses=5    # frames without a hand before smoothing resets
    ):
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.mp_draw = mp.solutions.drawing_utils

        self.smoothing = smoothing
        self.reset_after_misses = reset_after_misses
        self._smooth_pt = None
        self._misses = 0

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def best_hand(self, results, min_score=0.8):
        """
        Returns (hand_landmarks, score) for the most confident hand,
        or (None, 0.0) if nothing passes the gate.
        """
        if not results or not results.multi_hand_landmarks:
            self._register_miss()
            return None, 0.0

        best_lm, best_score = None, 0.0

        handedness = getattr(results, "multi_handedness", None)
        for i, lm in enumerate(results.multi_hand_landmarks):
            score = 1.0
            if handedness and i < len(handedness):
                try:
                    score = float(handedness[i].classification[0].score)
                except (AttributeError, IndexError):
                    score = 1.0
            if score > best_score:
                best_lm, best_score = lm, score

        if best_lm is None or best_score < min_score:
            self._register_miss()
            return None, best_score

        self._misses = 0
        return best_lm, best_score

    def draw_landmarks(self, frame, hand_landmarks):
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

    def get_fingertip(self, frame, hand_landmarks, smooth=True):
        """Index fingertip (landmark 8) in pixel coords, optionally smoothed."""
        h, w, _ = frame.shape

        tip = hand_landmarks.landmark[8]
        x = int(tip.x * w)
        y = int(tip.y * h)

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        if not smooth or self.smoothing <= 0:
            self._smooth_pt = (float(x), float(y))
            return x, y

        if self._smooth_pt is None:
            self._smooth_pt = (float(x), float(y))
        else:
            a = self.smoothing
            sx, sy = self._smooth_pt
            self._smooth_pt = (a * sx + (1 - a) * x, a * sy + (1 - a) * y)

        return int(self._smooth_pt[0]), int(self._smooth_pt[1])

    def _register_miss(self):
        self._misses += 1
        if self._misses >= self.reset_after_misses:
            self._smooth_pt = None

    def reset_smoothing(self):
        self._smooth_pt = None
        self._misses = 0

    def close(self):
        self.hands.close()