"""Self-check: every dictation leaves evidence — a history.log line and a
last-session.wav — so 'did it hear me?' is always answerable.

Run with the app venv: .venv/bin/python3 test_history_log.py
"""
import tempfile
import wave
from pathlib import Path

import numpy as np

import dictation_app as da


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        da.DATA_DIR = tmp
        da.HISTORY_FILE = tmp / "history.log"
        da.LAST_SESSION_WAV = tmp / "last-session.wav"

        da.log_history("typed: hello world")
        da.log_history("--- session end (1.5s audio captured) ---")
        lines = da.HISTORY_FILE.read_text().splitlines()
        assert len(lines) == 2, lines
        assert lines[0].endswith("typed: hello world"), lines[0]

        # 1 second of 440 Hz sine in 0.1s chunks, round-trips through the wav
        tone = da._generate_tone(440, 1.0, 0.5)
        chunks = [tone[i:i + da.CHUNK_SAMPLES]
                  for i in range(0, len(tone), da.CHUNK_SAMPLES)]
        da.save_session_wav(chunks)
        with wave.open(str(da.LAST_SESSION_WAV)) as w:
            assert w.getframerate() == da.SAMPLE_RATE
            assert w.getnframes() == len(tone), (w.getnframes(), len(tone))
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        assert np.abs(pcm / 32767 - tone).max() < 0.001

        da.save_session_wav([])  # empty session: no crash, no file overwrite

    print("OK: history log + session wav self-check passed")


if __name__ == "__main__":
    main()
