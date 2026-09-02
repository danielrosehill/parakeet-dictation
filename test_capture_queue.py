"""Self-check: a slow transcription must not lose mic audio.

The blocking InputStream.read() path holds only ~0.4s of audio and discards
overrun silently (overflowed stays False).  The capture must therefore run
via the PortAudio callback into a queue, so decode stalls lose nothing.

Run with the app venv: .venv/bin/python3 test_capture_queue.py
"""
import sys
import threading
import time

import numpy as np

import dictation_app as da

CHUNK = 1600
N_CHUNKS = 30


class FakeInputStream:
    """Callback-only mic: 30 sequential ramp chunks, fast.  No read() —
    the lossy blocking API must not be used at all."""

    def __init__(self, *_, callback=None, **kw):
        self._callback = callback
        self.produced = []
        self.active = False

    def start(self):
        self.active = True

        def _produce():
            time.sleep(0.05)  # let _begin_capture flip routing first
            for n in range(N_CHUNKS):
                data = np.arange(n * CHUNK, (n + 1) * CHUNK,
                                 dtype=np.float32).reshape(-1, 1)
                self.produced.extend(data.reshape(-1).tolist())
                self._callback(data, CHUNK, None, None)
                time.sleep(0.002)
        self._thread = threading.Thread(target=_produce, daemon=True)
        self._thread.start()

    def close(self):
        self.active = False


class RecordingVad:
    """Records every sample; emits a segment every 5 chunks."""

    def __init__(self):
        self.received = []
        self._segments = []
        self._since_emit = 0

    def accept_waveform(self, samples):
        self.received.extend(samples)
        self._since_emit += 1
        if self._since_emit == 5:
            self._since_emit = 0
            self._segments.append(da._SpeechSegment([0.0] * CHUNK))

    def is_speech_detected(self):
        return False

    def empty(self):
        return not self._segments

    @property
    def front(self):
        return self._segments[0]

    def pop(self):
        self._segments.pop(0)

    def flush(self):
        pass


class SlowRecognizer:
    """Each decode stalls 0.15s — far beyond the 0.4s mic buffer in total."""

    class _Stream:
        class _R:
            text = ""
        result = _R()

        def accept_waveform(self, sr, samples):
            pass

    def create_stream(self):
        return self._Stream()

    def decode_stream(self, s):
        time.sleep(0.15)


class TestEngine(da.ASREngine):
    def __init__(self, config):
        self._test_vad = RecordingVad()
        super().__init__(config, {"streaming": False},
                         on_text=lambda t: None, on_partial=lambda t: None,
                         on_error=lambda m: print("ERR:", m, file=sys.stderr))

    def _warm(self):
        pass

    def _ensure_models(self):
        pass

    def _get_recognizer(self):
        return SlowRecognizer()

    def _build_vad(self):
        return self._test_vad


def main():
    fakes = []
    real = da.sd.InputStream
    da.sd.InputStream = lambda *a, **kw: fakes.append(FakeInputStream(*a, **kw)) or fakes[-1]
    try:
        config = da.AppConfig(beep_volume=0.0)
        engine = TestEngine(config)
        engine.start()
        deadline = time.monotonic() + 15
        expected = N_CHUNKS * CHUNK
        while time.monotonic() < deadline:
            if len(engine._test_vad.received) >= expected:
                break
            time.sleep(0.05)
        engine.stop()
    finally:
        da.sd.InputStream = real

    got = engine._test_vad.received
    produced = fakes[0].produced
    assert len(got) == len(produced), (
        f"lost audio during slow decode: got {len(got)} of {len(produced)} samples")
    assert got == produced, "audio arrived corrupted or out of order"
    print("OK: no audio lost while transcription stalls the loop")


if __name__ == "__main__":
    sys.exit(main())
