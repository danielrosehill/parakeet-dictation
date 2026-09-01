"""Self-check: Gemini Live events route into the same partial/final
callbacks the local streaming path uses.

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

    eng._handle_gemini_event(None, True)
    eng._handle_gemini_event(NS(interim_input_transcription=None,
                                input_transcription=None), True)
    assert got == [], "empty events must be ignored"

    eng._handle_gemini_event(NS(interim_input_transcription=NS(text="hello wor "),
                                input_transcription=None), True)
    assert got == [("status", "hello wor"), ("partial", "hello wor")], got

    got.clear()
    eng._handle_gemini_event(NS(interim_input_transcription=None,
                                input_transcription=NS(text="Hello world.")), True)
    assert got == [("commit", "Hello world.")], got

    # Without partial-overwrite, finals are typed whole and interims only show
    got.clear()
    eng._handle_gemini_event(NS(interim_input_transcription=NS(text="a"),
                                input_transcription=NS(text="A.")), False)
    assert got == [("status", "a"), ("text", "A.")], got

    # A stop() bump invalidates queued partials
    got.clear()
    eng._partial_seq += 1
    eng._emit_partial_type(eng._partial_seq - 1, "stale")
    assert got == [], "stale partial must not be typed"

    # PCM conversion: full-scale float -> int16 little-endian, clipped
    pcm = da.pcm16([1.0, -1.0, 2.0, 0.0])
    assert pcm == b"\xff\x7f\x01\x80\xff\x7f\x00\x00", pcm

    # Cloud profiles never count as "downloaded" for the first-run dialog
    assert not da._any_model_downloaded({"gemini-live": {"backend": "gemini"}})

    print("OK: gemini events route to partial/commit/final callbacks")


if __name__ == "__main__":
    sys.exit(main())
