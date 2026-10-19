import threading
import queue
import time
import sys

import pyttsx3

# Windows COM init -- pyttsx3 does NOT do this for you when the engine runs in a
# background thread. Without it, SAPI goes silent after the first utterance.
try:
    import pythoncom          # ships with pywin32
    _HAS_PYTHONCOM = True
except ImportError:
    _HAS_PYTHONCOM = False

try:
    import win32com.client    # ships with pywin32
    _HAS_WIN32COM = True
except ImportError:
    _HAS_WIN32COM = False


class VoiceAssistant:
    """
    Thread-safe, non-blocking text-to-speech helper.

    IMPORTANT (why this file looks the way it does):
    A pyttsx3 engine on Windows SAPI5 can only be driven through runAndWait()
    reliably ONCE. After the first call the driver's event loop is torn down,
    and every later say() is accepted silently but never spoken -- the log
    prints, the speakers stay quiet. The fix is to build a fresh engine for
    each utterance and destroy it afterwards, so pyttsx3's internal engine
    cache releases it.

    backend:
        "auto"    -> use SAPI directly if pywin32 is available, else pyttsx3
        "sapi"    -> force direct SAPI (fastest and most reliable on Windows)
        "pyttsx3" -> force pyttsx3 (works on Linux/macOS too)
    """

    def __init__(self, rate=185, repeat_interval=1.5, voice_index=2,
                 volume=1.0, backend="auto"):
        self.rate = rate
        self.volume = volume
        self.repeat_interval = repeat_interval
        self.voice_index = voice_index

        if backend == "auto":
            backend = "sapi" if (_HAS_WIN32COM and sys.platform == "win32") else "pyttsx3"
        self.backend = backend

        self._queue = queue.Queue()
        self._idle = threading.Event()
        self._idle.set()
        self._speaking = False
        self._sapi = None          # reused SpVoice object (safe to reuse)

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        print(f"[VOICE] Threaded voice assistant initialized (backend={self.backend}).")

    # ------------------------------------------------------------------
    # backend: direct SAPI via pywin32
    # ------------------------------------------------------------------

    def _sapi_init(self):
        sapi = win32com.client.Dispatch("SAPI.SpVoice")

        # SAPI rate is -10..10; pyttsx3 rate is words-per-minute around 200.
        sapi.Rate = int(max(-10, min(10, round((self.rate - 200) / 20.0))))
        sapi.Volume = int(max(0, min(100, round(self.volume * 100))))

        try:
            voices = sapi.GetVoices()
            count = voices.Count
            if count:
                idx = self.voice_index if 0 <= self.voice_index < count else 0
                sapi.Voice = voices.Item(idx)
                print(f"[VOICE] Using voice: {voices.Item(idx).GetDescription()}")
        except Exception as e:
            print(f"[VOICE] Could not set voice index: {e}")

        return sapi

    def _sapi_speak(self, text):
        if self._sapi is None:
            self._sapi = self._sapi_init()
        self._sapi.Speak(text)     # synchronous; returns when speech ends

    # ------------------------------------------------------------------
    # backend: pyttsx3 (fresh engine per utterance -- this is the fix)
    # ------------------------------------------------------------------

    def _pyttsx3_speak(self, text):
        engine = pyttsx3.init()
        try:
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)

            voices = engine.getProperty('voices')
            if voices:
                idx = self.voice_index if 0 <= self.voice_index < len(voices) else 0
                engine.setProperty('voice', voices[idx].id)

            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass
            # Dropping the reference lets pyttsx3 release its cached engine,
            # so the NEXT init() genuinely builds a new one.
            del engine

    # ------------------------------------------------------------------
    # worker
    # ------------------------------------------------------------------

    def _worker(self):
        if _HAS_PYTHONCOM:
            try:
                pythoncom.CoInitialize()
            except Exception as e:
                print(f"[VOICE] CoInitialize skipped: {e}")

        last_text = None
        last_time = 0.0

        while True:
            text = self._queue.get()
            if text is None:
                break

            self._idle.clear()
            now = time.time()

            # Don't let the same line be spoken 30 times a second
            if text == last_text and (now - last_time) < self.repeat_interval:
                if self._queue.empty():
                    self._idle.set()
                continue

            last_text = text
            last_time = now
            self._speaking = True
            print(f"[ANVI VOICE] {text}")

            try:
                if self.backend == "sapi":
                    self._sapi_speak(text)
                else:
                    self._pyttsx3_speak(text)
            except Exception as e:
                print(f"[VOICE ERROR] {e}")
                self._sapi = None
                # One retry on the other backend, so a demo never goes mute
                try:
                    if self.backend == "sapi":
                        self._pyttsx3_speak(text)
                    elif _HAS_WIN32COM:
                        self._sapi_speak(text)
                except Exception as e2:
                    print(f"[VOICE FATAL] {e2}")
            finally:
                self._speaking = False
                if self._queue.empty():
                    self._idle.set()

        if _HAS_PYTHONCOM:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # public API (unchanged -- main.py needs no edits)
    # ------------------------------------------------------------------

    def speak(self, text):
        """Queue text, dropping stale queued lines so audio never lags behind."""
        if not text:
            return
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._idle.clear()
        self._queue.put(text)

    def say_and_wait(self, text, timeout=15.0):
        self.speak(text)
        self.wait_until_idle(timeout)

    def wait_until_idle(self, timeout=15.0):
        time.sleep(0.05)
        self._idle.wait(timeout)

    @property
    def is_speaking(self):
        return self._speaking

    def stop(self):
        self._queue.put(None)


if __name__ == "__main__":
    print("================================")
    print("      ANVI VOICE TEST")
    print("================================")

    v = VoiceAssistant(rate=185, voice_index=2)

    v.say_and_wait("One. This is ANVI.")
    v.say_and_wait("Two. Move your hand to the right.")
    v.say_and_wait("Three. Move up.")
    v.say_and_wait("Four. Button reached.")
    v.say_and_wait("Five. If you heard all five, the fix is working.")

    v.stop()
    time.sleep(0.5)
    print("Voice test completed.")