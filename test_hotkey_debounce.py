"""Self-check: a held hotkey must activate exactly once.

X11 auto-repeat re-delivers a held key as release+press pulses (~33/s
after a ~0.5s delay) and pynput re-activates the hotkey on every pulse.
Each re-fire used to toggle dictation off again right after it started —
the session died before the first word.  Two guards fix it:

1. HotkeyManager._debounce — only the first pulse of a train fires; every
   suppressed pulse re-arms the window, so a hold of any length is one
   activation.
2. ASREngine.start() flips is_running before it returns, so a fire that
   does slip through maps to stop() instead of starting a second thread.

Run with the app venv: .venv/bin/python3 test_hotkey_debounce.py
"""
import sys
import threading
import time

import dictation_app as da

failures = []


def check(tag, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {tag} {detail}", flush=True)
    if not ok:
        failures.append(tag)


# --- 1. debounce: auto-repeat pulse train fires once -----------------------

GAP = 0.2
count = [0]
wrapper = da.HotkeyManager._debounce(lambda: count.__setitem__(0, count[0] + 1),
                                     gap=GAP)

wrapper()                       # press
for _ in range(6):              # auto-repeat pulses, 30ms apart
    time.sleep(0.03)
    wrapper()
check("single press+repeats -> one fire", count[0] == 1, f"count={count[0]}")

time.sleep(GAP + 0.05)
wrapper()                       # a genuine new press after the gap
check("new press after gap fires", count[0] == 2, f"count={count[0]}")

time.sleep(GAP + 0.05)
wrapper()                       # press...
for _ in range(20):             # ...held well past the gap: 0.6s of pulses
    time.sleep(0.03)
    wrapper()
check("long hold still one fire", count[0] == 3, f"count={count[0]}")


# --- 2. is_running is true the moment start() returns ----------------------

class SlowInitEngine(da.ASREngine):
    """Engine whose init is slow, like a real mic/model open."""

    def __init__(self):
        self.ran = threading.Event()
        super().__init__(da.AppConfig(beep_volume=0.0), {"streaming": False},
                         on_text=lambda t: None, on_partial=lambda t: None,
                         on_error=lambda m: None)

    def _warm(self):
        pass

    def _ensure_models(self):
        time.sleep(0.2)         # the window where toggle() used to misread

    def _run_offline(self):
        self.ran.set()
        self._stop_event.wait(timeout=5)


engine = SlowInitEngine()
engine.start()
check("is_running true right after start()", engine.is_running)
engine.ran.wait(timeout=5)
engine.stop()
check("is_running false after stop()", not engine.is_running)

# A start that dies (missing models) must clear is_running again.


class FailingEngine(SlowInitEngine):
    def _ensure_models(self):
        raise FileNotFoundError("no models in test")


fail_engine = FailingEngine()
fail_engine.start()
fail_engine._thread.join(timeout=5)
check("is_running false after failed start", not fail_engine.is_running)

print("RESULT:", "FAIL " + ",".join(failures) if failures else "ALL PASS",
      flush=True)
sys.exit(1 if failures else 0)
