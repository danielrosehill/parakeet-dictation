"""Self-check for HotkeyCaptureButton using real XTEST key events.

Run with the app venv: .venv/bin/python3 test_hotkey_capture.py
Needs Xvfb and xdotool; skips (exit 0) if either is missing.
"""
import atexit
import os
import shutil
import subprocess
import sys
import threading
import time

if not (shutil.which("Xvfb") and shutil.which("xdotool")):
    print("SKIP: Xvfb/xdotool not installed")
    sys.exit(0)

DISPLAY = ":97"
xvfb = subprocess.Popen(["Xvfb", DISPLAY, "-screen", "0", "640x480x24"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
atexit.register(xvfb.terminate)
os.environ["DISPLAY"] = DISPLAY
time.sleep(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib  # noqa: E402

import dictation_app as da  # noqa: E402

win = Gtk.Window(title="hktest-window")
btn = da.HotkeyCaptureButton("<ctrl>+0")
win.add(btn)
win.show_all()

failures = []


def xdo(*args):
    subprocess.run(["xdotool", *args], check=True)


def check(tag, expect_binding, expect_capturing):
    ok = btn.binding == expect_binding and btn._capturing == expect_capturing
    print(f"{'PASS' if ok else 'FAIL'} {tag}: binding={btn.binding!r} "
          f"capturing={btn._capturing}", flush=True)
    if not ok:
        failures.append(tag)


def start_capture():
    GLib.idle_add(btn._on_clicked, None)
    time.sleep(0.3)


def worker():
    time.sleep(0.5)
    xdo("search", "--name", "hktest-window", "windowfocus")
    time.sleep(0.3)

    start_capture()
    xdo("keydown", "ctrl")
    time.sleep(0.2)
    check("ctrl-down keeps waiting", "<ctrl>+0", True)
    xdo("key", "e")
    time.sleep(0.2)
    xdo("keyup", "ctrl")
    time.sleep(0.2)
    check("ctrl+e", "<ctrl>+e", False)

    start_capture()
    xdo("keydown", "super")
    time.sleep(0.2)
    xdo("key", "e")
    time.sleep(0.2)
    xdo("keyup", "super")
    time.sleep(0.2)
    check("super+e", "<cmd>+e", False)

    start_capture()
    xdo("key", "F9")
    time.sleep(0.2)
    check("bare F9 bracketed", "<f9>", False)

    start_capture()
    xdo("key", "Pause")
    time.sleep(0.2)
    check("bare Pause bracketed", "<pause>", False)

    start_capture()
    xdo("key", "Return")
    time.sleep(0.2)
    check("Return maps to enter", "<enter>", False)

    start_capture()
    xdo("key", "Escape")
    time.sleep(0.2)
    check("Escape cancels", "<enter>", False)

    start_capture()
    xdo("key", "ctrl")
    time.sleep(0.2)
    xdo("key", "e")
    time.sleep(0.2)
    check("tap-ctrl then e -> bare e", "e", False)

    GLib.idle_add(Gtk.main_quit)


threading.Thread(target=worker, daemon=True).start()
Gtk.main()
print("RESULT:", "FAIL " + ",".join(failures) if failures else "ALL PASS", flush=True)
sys.exit(1 if failures else 0)
