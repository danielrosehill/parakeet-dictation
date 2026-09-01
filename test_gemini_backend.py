"""Self-check: Gemini Live interims reach the status bar only and
finals are typed whole.

Run with the app venv: .venv/bin/python3 test_gemini_backend.py
Pure logic — no display, audio, network, or API key needed.
"""
import sys
from types import SimpleNamespace as NS

import dictation_app as da


def main():
    da.ASREngine._warm = lambda self: None
    da.GLib.idle_add = lambda fn, *a: fn(*a)  # run callbacks inline
    got = []
    eng = da.ASREngine(
        da.AppConfig(beep_volume=0.0), {"backend": "gemini"},
        on_text=lambda t: got.append(("text", t)),
        on_partial=lambda t: got.append(("status", t)),
        on_error=lambda t: got.append(("error", t)),
        on_partial_type=lambda t: got.append(("partial", t)),
        on_commit_partial=lambda t: got.append(("commit", t)),
    )

    eng._handle_gemini_event(None)
    eng._handle_gemini_event(NS(interim_input_transcription=None,
                                input_transcription=None))
    assert got == [], "empty events must be ignored"

    # Interims only update the status bar — never typed
    eng._handle_gemini_event(NS(interim_input_transcription=NS(text="hello wor "),
                                input_transcription=None))
    assert got == [("status", "hello wor")], got

    # Finals are typed whole
    got.clear()
    eng._handle_gemini_event(NS(interim_input_transcription=NS(text="a"),
                                input_transcription=NS(text="Hello world.")))
    assert got == [("status", "a"), ("text", "Hello world.")], got

    # PCM conversion: full-scale float -> int16 little-endian, clipped
    pcm = da.pcm16([1.0, -1.0, 2.0, 0.0])
    assert pcm == b"\xff\x7f\x01\x80\xff\x7f\x00\x00", pcm

    # Cloud profiles never count as "downloaded" for the first-run dialog
    assert not da._any_model_downloaded({"gemini-live": {"backend": "gemini"}})

    print("OK: gemini interims go to status, finals are typed whole")


if __name__ == "__main__":
    sys.exit(main())
