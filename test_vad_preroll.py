"""Self-check for VAD pre-roll: leading audio before detection is kept.

Run with the app venv: .venv/bin/python3 test_vad_preroll.py
Uses a scripted fake in place of the native TEN VAD — pure logic.
"""
import sys

import dictation_app as da

HOP = 256


class FakeVad:
    """Returns scripted speech probabilities, one per hop."""
    def __init__(self, probs):
        self._probs = list(probs)

    def process(self, chunk):
        return self._probs.pop(0), 0


def detector(probs, **kw):
    det = da.TenVadDetector(min_speech_duration=0.01, min_silence_duration=0.02,
                            **kw)
    det._vad = FakeVad(probs)
    return det


def feed(det, values):
    for v in values:
        det.accept_waveform([v] * HOP)


def main():
    # 5 silence hops, 3 speech, 2 silence (>= min_silence -> segment emits)
    det = detector([0.0] * 5 + [1.0] * 3 + [0.0] * 2)
    feed(det, [0.1] * 5 + [0.9] * 3 + [0.1] * 2)
    assert not det.empty(), "segment should have been emitted"
    seg = det.front.samples
    # Pre-roll (5 hops of 0.1) + speech (3) + trailing silence (2) = 10 hops
    assert len(seg) == 10 * HOP, f"expected 10 hops, got {len(seg) / HOP}"
    assert seg[0] == 0.1, "segment must start with pre-roll audio"
    assert seg[5 * HOP] == 0.9, "speech audio must follow pre-roll"

    # Pre-roll ring is bounded to ~0.3s (18 hops at 16kHz)
    det = detector([0.0] * 40 + [1.0] * 3 + [0.0] * 2)
    feed(det, [0.1] * 40 + [0.9] * 3 + [0.1] * 2)
    assert len(det.front.samples) == (18 + 3 + 2) * HOP, "pre-roll must be bounded"

    # A 1-hop blip must NOT become a segment (gate counts speech, not buffer)
    det = detector([0.0] * 5 + [1.0] + [0.0] * 3)
    det._min_speech_samples = 4 * HOP
    feed(det, [0.1] * 5 + [0.9] + [0.1] * 3)
    assert det.empty(), "blip inflated by pre-roll must not qualify as speech"

    print("OK: all VAD pre-roll checks passed")


if __name__ == "__main__":
    sys.exit(main())
