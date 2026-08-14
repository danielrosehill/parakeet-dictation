"""Self-check: the mic stays open across sessions and pre-press audio is kept.

Opening the input stream on each hotkey press cost 100-300 ms, and a cold
recognizer fetch could add seconds — speech in that window was never
captured.  The engine must open the mic once at build, keep the newest
0.5 s in a ring while idle, and splice that ring in front of the live
queue when capture starts.

Run with the app venv: .venv/bin/python3 test_mic_preroll.py
Pure logic — no display, audio device, or model files needed.
"""
import sys
import time

import numpy as np

import dictation_app as da

CHUNK = da.CHUNK_SAMPLES


class FakeInputStream:
    """Manual-feed mic: the test pushes chunks through the callback."""

    def __init__(self, *_, callback=None, **kw):
        self._callback = callback
        self.started = False
        self.closed = False

    @property
    def active(self):
        return self.started and not self.closed

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def feed(self, n: int):
        data = np.arange(n * CHUNK, (n + 1) * CHUNK,
                         dtype=np.float32).reshape(-1, 1)
        self._callback(data, CHUNK, None, None)


class RecordingVad:
    def __init__(self):
        self.received = []

    def accept_waveform(self, samples):
        self.received.extend(samples)

    def is_speech_detected(self):
        return False

    def empty(self):
        return True

    def flush(self):
        pass


class FastRecognizer:
    class _Stream:
        class _R:
            text = ""
        result = _R()

        def accept_waveform(self, sr, samples):
            pass

    def create_stream(self):
        return self._Stream()

    def decode_stream(self, s):
        pass


class TestEngine(da.ASREngine):
    """Real _warm, _ensure_mic and run loop; only model access is stubbed."""

    def __init__(self, config):
        self._test_vad = RecordingVad()
        super().__init__(config, {"streaming": False},
                         on_text=lambda t: None, on_partial=lambda t: None,
                         on_error=lambda m: print("ERR:", m, file=sys.stderr))

    def _ensure_models(self):
        pass

    def _get_recognizer(self):
        return FastRecognizer()

    def _build_vad(self):
        return self._test_vad


def expected(lo: int, hi: int) -> list:
    return list(np.arange(lo * CHUNK, hi * CHUNK, dtype=np.float32))


def wait_for(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.01)
    return cond()


def main():
    fakes = []
    real = da.sd.InputStream
    da.sd.InputStream = (
        lambda *a, **kw: fakes.append(FakeInputStream(*a, **kw)) or fakes[-1])
    try:
        engine = TestEngine(da.AppConfig(beep_volume=0.0))

        # Mic opens at engine build (via _warm), before any hotkey press
        assert wait_for(lambda: fakes), "mic not opened at engine build"
        mic = fakes[0]
        assert mic.started, "mic stream not started at build"

        # 1) 10 idle chunks before the press; ring keeps only the last 5
        for n in range(10):
            mic.feed(n)
        engine.start()
        assert wait_for(lambda: engine._capturing), "capture never began"
        for n in range(10, 15):
            mic.feed(n)
        want = expected(5, 15)
        wait_for(lambda: len(engine._test_vad.received) >= len(want))
        engine.stop()
        assert engine._test_vad.received == want, (
            f"pre-press audio lost or reordered: got "
            f"{len(engine._test_vad.received)} samples, want {len(want)} "
            f"starting at chunk 5")

        # 2) stop leaves the mic open; idle audio goes to the ring
        assert not mic.closed, "mic closed on stop — must stay open"
        for n in range(15, 17):
            mic.feed(n)
        assert engine._test_vad.received == want, (
            "idle audio leaked into a stopped session")

        # 3) the next session starts with those ring chunks
        engine._test_vad.received.clear()
        engine.start()
        assert wait_for(lambda: engine._capturing), "second capture never began"
        want2 = expected(15, 17)
        wait_for(lambda: len(engine._test_vad.received) >= len(want2))
        engine.stop()
        assert engine._test_vad.received == want2, (
            "pre-press ring not spliced into the next session")
        assert not mic.closed, "mic closed between sessions"

        # 4) a device change in config reopens the stream
        engine._config.audio_device = "3"
        engine.start()
        assert wait_for(lambda: len(fakes) >= 2), (
            "device change did not reopen the mic")
        engine.stop()
        assert mic.closed, "old stream left open after device change"

        # 5) close() releases the device (engine rebuild path)
        engine.close()
        assert fakes[1].closed, "close() left the mic open"
    finally:
        da.sd.InputStream = real

    print("OK: mic persists across sessions and pre-press audio is captured")


if __name__ == "__main__":
    sys.exit(main())
