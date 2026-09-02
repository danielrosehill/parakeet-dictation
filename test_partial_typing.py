"""Self-check for partial-overwrite typing: diff-typing and stale-partial drop.

Run with the app venv: .venv/bin/python3 test_partial_typing.py
Pure logic — no display or audio needed.
"""
import sys

import dictation_app as da


def recording_typer():
    """TextTyper whose key/type operations are recorded, not executed."""
    t = da.TextTyper.__new__(da.TextTyper)  # skip __init__'s healing keyup
    t._method = "xdotool"
    t._partial = ""
    ops = []
    t._send_backspaces = lambda n: n and ops.append(("bs", n))
    t._type_raw = lambda s: ops.append(("type", s))
    t._wait_modifiers_up = lambda timeout=2.0: True
    return t, ops


def main():
    # -- _diff -------------------------------------------------------------
    assert da.TextTyper._diff("", "Hello") == (0, "Hello")
    assert da.TextTyper._diff("Hel", "Hello") == (0, "lo")
    assert da.TextTyper._diff("Hello there", "Hello") == (6, "")
    assert da.TextTyper._diff("Hello wold", "Hello world") == (2, "rld")
    assert da.TextTyper._diff("same", "same") == (0, "")

    # -- type_partial types only the changed tail --------------------------
    t, ops = recording_typer()
    t.type_partial("Hel")
    t.type_partial("Hello wor")
    t.type_partial("Hello wor")          # unchanged revision: no ops
    t.type_partial("Hello world")
    assert ops == [("type", "Hel"), ("type", "lo wor"), ("type", "ld")], ops

    # -- revision that rewrites the middle backspaces just the tail --------
    t, ops = recording_typer()
    t.type_partial("Hello wold")
    t.type_partial("Hello world")
    assert ops == [("type", "Hello wold"), ("bs", 2), ("type", "rld")], ops

    # -- commit fixes the tail and adds the trailing space -----------------
    t, ops = recording_typer()
    t.type_partial("Hello ther")
    t.commit_partial("Hello there.")
    assert ops == [("type", "Hello ther"), ("type", "e. ")], ops
    assert t._partial == ""

    # -- empty commit erases the whole partial -----------------------------
    t, ops = recording_typer()
    t.type_partial("oops")
    t.commit_partial("")
    assert ops == [("type", "oops"), ("bs", 4)], ops

    # -- stale queued partials are dropped; stop() invalidates all ---------
    typed = []
    eng = da.ASREngine(da.AppConfig(), {}, on_text=lambda t: None,
                       on_partial=lambda t: None, on_error=lambda m: None,
                       on_partial_type=typed.append)
    eng._partial_seq = 5
    eng._emit_partial_type(3, "stale")     # older than newest -> dropped
    eng._emit_partial_type(5, "current")   # newest -> typed
    assert typed == ["current"], typed
    eng._running = True
    eng.stop()                              # bumps seq
    eng._emit_partial_type(5, "after stop")
    assert typed == ["current"], typed

    print("OK: all partial-typing checks passed")


if __name__ == "__main__":
    sys.exit(main())
