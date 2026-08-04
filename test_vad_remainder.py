"""Self-check: no audio is lost at accept_waveform() chunk boundaries.

The app feeds 1600-sample chunks; the VAD consumes 256-sample hops, so a
remainder carries between calls.  The float audio kept for ASR must stay
byte-identical and gap-free with what the VAD classified.

Run with the app venv: .venv/bin/python3 test_vad_remainder.py
"""
import sys

import dictation_app as da

HOP = 256
CHUNK = 1600  # 0.1s at 16kHz, same as the capture loop


class FakeVad:
    def __init__(self, probs):
        self._probs = list(probs)

    def process(self, chunk):
        return self._probs.pop(0), 0


def main():
    det = da.TenVadDetector(min_speech_duration=0.01, min_silence_duration=0.02)
    det._vad = FakeVad([1.0] * 62 + [0.0] * 40)

    # 1 second of a ramp 1..16000, fed in app-sized chunks, all speech
    ramp = [float(v) for v in range(1, 16001)]
    for i in range(0, 16000, CHUNK):
        det.accept_waveform(ramp[i:i + CHUNK])
    # then silence until the segment emits
    for _ in range(2):
        det.accept_waveform([0.0] * CHUNK)

    assert not det.empty(), "segment should have been emitted"
    seg = det.front.samples
    got = seg[:16000]
    if got != ramp:
        for j, (a, b) in enumerate(zip(got, ramp)):
            if a != b:
                raise AssertionError(
                    f"audio lost at sample {j}: got {a}, expected {b} "
                    f"(segment holds {len(seg)} samples)")
        raise AssertionError(f"segment too short: {len(seg)} < 16000")

    print("OK: chunk-boundary audio is contiguous")


if __name__ == "__main__":
    sys.exit(main())
