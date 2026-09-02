"""Self-check for personal-vocabulary word replacements.

Run with the app venv: .venv/bin/python3 test_word_replacements.py
Pure logic — no display or audio needed.
"""
import dictation_app as da


def main():
    m = {"herder": "herdr"}

    # whole-word swap, punctuation kept
    assert da.apply_word_replacements("the herder walks", m) == "the herdr walks"
    assert da.apply_word_replacements("Nice herder.", m) == "Nice herdr."

    # leading capital preserved (sentence start)
    assert da.apply_word_replacements("Herder is here", m) == "Herdr is here"

    # case-insensitive match
    assert da.apply_word_replacements("HERDER", m) == "Herdr"

    # no substring hits inside longer words
    assert da.apply_word_replacements("shepherder", m) == "shepherder"

    # multi-word key, longest-first wins
    m2 = {"note": "n0te", "note book": "notebook"}
    assert da.apply_word_replacements("my note book", m2) == "my notebook"
    assert da.apply_word_replacements("my note", m2) == "my n0te"

    # empty mapping / empty text are no-ops
    assert da.apply_word_replacements("hello", {}) == "hello"
    assert da.apply_word_replacements("", m) == ""

    # settings-line parsing shape: junk lines are skipped
    raw = "herder=herdr\n\nbadline\n =x\n a = b "
    parsed = {
        k.strip(): v.strip()
        for k, _, v in (line.partition("=") for line in raw.splitlines())
        if k.strip() and v.strip()
    }
    assert parsed == {"herder": "herdr", "a": "b"}

    print("OK")


if __name__ == "__main__":
    main()
