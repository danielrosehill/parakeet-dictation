"""Self-check: switching model while idle rebuilds the ASR engine.

Every call site mutates the controller's shared AppConfig object before
calling apply_config, so a profile-diff against that same object never
fired and the engine kept the previous model's profile.

Run with the app venv: .venv/bin/python3 test_model_switch.py
Pure logic — no display, audio, or model files needed.
"""
import sys
import tempfile
from pathlib import Path

import dictation_app as da


def main():
    # Keep the test off the real config file and skip model warm-up
    tmp = Path(tempfile.mkdtemp())
    da.CONFIG_DIR = tmp
    da.CONFIG_FILE = tmp / "config.json"
    da.ASREngine._warm = lambda self: None

    c = da.DictationController(da.AppConfig(beep_volume=0.0))
    first = c._engine

    c.switch_model("streaming")
    assert c._engine is not first, "engine not rebuilt on model switch"
    assert c._engine._profile is c.profiles["streaming"], "stale profile"

    # The aliased-object path the settings dialog uses
    cfg = c.config
    cfg.model_profile = "laptop"
    c.apply_config(cfg)
    assert c._engine._profile is c.profiles["laptop"], "aliased change missed"

    # No profile change while idle: keep the warmed engine
    engine = c._engine
    c.apply_config(c.config)
    assert c._engine is engine, "engine rebuilt without a profile change"

    print("OK: model switch while idle rebuilds the engine")


if __name__ == "__main__":
    sys.exit(main())
