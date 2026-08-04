"""Self-check for the modifier guard around keystroke injection.

A modifier held while xdotool types turns every character into a hotkey
chord (held Ctrl made each 't' a Ctrl+T — opened a pile of terminals).
Verifies: piece-wise typing, drop-on-held, phantom-latch healing, and
the backspace-burst guard.

Run with the app venv: .venv/bin/python3 test_typing_guard.py
Pure logic — no display or audio needed.
"""
import sys

import dictation_app as da


def guarded_typer(wait_results):
    """xdotool TextTyper with scripted _wait_modifiers_up results and
    subprocess.run recorded instead of executed."""
    t = da.TextTyper.__new__(da.TextTyper)  # skip __init__'s healing keyup
    t._method = "xdotool"
    t._partial = ""
    script = list(wait_results)
    t._wait_modifiers_up = lambda timeout=2.0: script.pop(0) if script else True
    calls = []
    da.subprocess.run = lambda cmd, **kw: calls.append(cmd)
    return t, calls


def typed(calls):
    return "".join(c[-1] for c in calls if c[:2] == ["xdotool", "type"])


def main():
    real_run = da.subprocess.run
    try:
        # -- clear modifiers: text goes out in <=24-char pieces, in order --
        text = "abcdefghijklmnopqrstuvwxyz0123456789"
        t, calls = guarded_typer([True, True])
        t._type_raw(text)
        pieces = [c[-1] for c in calls if c[:2] == ["xdotool", "type"]]
        assert pieces == [text[:24], text[24:]], pieces
        assert typed(calls) == text

        # -- modifier held from the start: nothing typed, latch healed ----
        t, calls = guarded_typer([False, False])
        t._type_raw("terminal terminal")
        assert typed(calls) == "", calls
        assert ["xdotool", "keyup", "ctrl", "shift", "alt", "super"] in calls

        # -- modifier pressed mid-injection: rest of the text dropped -----
        t, calls = guarded_typer([True, False, False])
        t._type_raw("x" * 30)
        assert typed(calls) == "x" * 24, calls

        # -- phantom latch: heal clears it, typing proceeds ---------------
        t, calls = guarded_typer([False, True])
        t._type_raw("ok")
        assert typed(calls) == "ok", calls
        assert ["xdotool", "keyup", "ctrl", "shift", "alt", "super"] in calls

        # -- backspace burst refused while a modifier is held -------------
        t, calls = guarded_typer([False, False])
        t._send_backspaces(5)
        assert not any("BackSpace" in c for c in calls), calls

        # -- backspace burst sent once modifiers are up -------------------
        t, calls = guarded_typer([True])
        t._send_backspaces(5)
        assert ["xdotool", "key", "--delay", "2", "--repeat", "5",
                "BackSpace"] in calls, calls
    finally:
        da.subprocess.run = real_run

    print("test_typing_guard: all checks passed")


if __name__ == "__main__":
    sys.exit(main())
