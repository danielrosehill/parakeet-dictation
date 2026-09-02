"""Self-check: long segments are decoded in pieces, and a blank piece is
retried instead of silently dropped.

Parakeet TDT emits zero tokens for long audio (fine to ~34s, blank from ~36s
up, in both the int8 and fp32 builds), and blanks sporadically below that.
So no single decode may exceed MAX_DECODE_SECS, the split must not lose a
sample, and a blank result must be retried before the words are thrown away.

Run with the app venv: .venv/bin/python3 test_decode_split.py
"""
import dictation_app as da

SR = da.SAMPLE_RATE


class FakeStream:
    def __init__(self, result_for):
        self._result_for = result_for
        self.samples = None

    def accept_waveform(self, _rate, samples):
        self.samples = list(samples)

    @property
    def result(self):
        return type("R", (), {"text": self._result_for(self.samples)})


class FakeRecognizer:
    """Transcribes only what the given rule accepts; everything else blank."""

    def __init__(self, rule):
        self._rule = rule
        self.decodes = 0

    def create_stream(self):
        return FakeStream(self._rule)

    def decode_stream(self, _stream):
        self.decodes += 1


def test_split_caps_length_and_keeps_every_sample():
    # 33.2s — the length from the real missed dictation — with quiet gaps
    # every 5s so there is somewhere sensible to cut.
    audio = []
    for i in range(332):
        loud = (i % 50) < 45
        audio += [0.5 if loud else 0.0] * (SR // 10)
    assert len(audio) == int(33.2 * SR)

    pieces = da.split_for_decode(audio)
    assert len(pieces) > 1, "33.2s must not go into one decode"
    cap = da.MAX_DECODE_SECS * 1.2 * SR
    for p in pieces:
        assert len(p) <= cap, f"piece of {len(p) / SR:.1f}s exceeds the cap"
    rejoined = [v for p in pieces for v in p]
    assert rejoined == audio, "split lost or reordered audio"

    # each cut lands in a silent gap, not mid-word
    at = 0
    for p in pieces[:-1]:
        at += len(p)
        assert audio[at] == 0.0, f"cut at {at / SR:.1f}s landed inside speech"


def test_short_segment_is_one_piece():
    audio = [0.5] * int(9.0 * SR)
    assert da.split_for_decode(audio) == [audio]


def test_blank_is_retried_before_being_dropped():
    engine = object.__new__(da.ASREngine)  # no __init__: it starts threads
    samples = [0.1] * (2 * SR)

    # This model blanks on the audio as given but decodes it with a tail of
    # silence appended — exactly the observed failure.
    padded = FakeRecognizer(lambda s: "recovered" if len(s) > len(samples) else "")
    assert engine._decode_piece(padded, samples) == "recovered"

    # ...and the gain retry catches what the padding retry does not.
    quieter = FakeRecognizer(lambda s: "recovered" if s and s[0] < 0.1 else "")
    assert engine._decode_piece(quieter, samples) == "recovered"

    # A genuinely silent piece still returns blank, after 3 tries, not text.
    silent = FakeRecognizer(lambda _s: "")
    assert engine._decode_piece(silent, samples) == ""
    assert silent.decodes == 3

    # Audio that decodes first time costs exactly one decode.
    good = FakeRecognizer(lambda _s: "hello")
    assert engine._decode_piece(good, samples) == "hello"
    assert good.decodes == 1


def main():
    test_split_caps_length_and_keeps_every_sample()
    test_short_segment_is_one_piece()
    test_blank_is_retried_before_being_dropped()
    print("OK: long segments split at quiet points, blanks retried")


if __name__ == "__main__":
    main()
