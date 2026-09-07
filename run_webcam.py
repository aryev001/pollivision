#!/usr/bin/env python3
"""Zero-configuration entry point: run PolliVision on this machine's webcam.

Clone the repository, open it in VS Code, press F5 (or run ``python
run_webcam.py``) and the perception stack runs live on the built-in camera.

This exists so that the first run needs no ``pip install -e .``, no activated
environment and no memory of the CLI's argument names. It puts the package on
``sys.path`` itself, checks the handful of things that actually go wrong on a
fresh machine, and explains each one in terms of what to type next rather than
raising a traceback from six frames inside OpenCV.

Anything this script accepts is passed straight through to
``pollivision webcam``, so once it works, these are the same:

    python run_webcam.py --fast --mirror
    pollivision webcam --fast --mirror
"""

from __future__ import annotations

import sys
from pathlib import Path

# The importable package lives one directory down from the repository root.
ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT / "pollivision" if (ROOT / "pollivision" / "pollivision").is_dir() else ROOT
sys.path.insert(0, str(PACKAGE_ROOT))

# Split in two, because `--help` and `--list-cameras` are exactly the commands
# someone runs while still sorting out their install, and they need neither the
# detector nor the vision-language model.
CAPTURE_DEPS = {
    "cv2": ("opencv-python", "camera capture and the preview window"),
    "numpy": ("numpy", "array handling"),
    "yaml": ("PyYAML", "configuration files"),
}
MODEL_DEPS = {
    "torch": ("torch", "the detector and the vision-language model"),
    "ultralytics": ("ultralytics", "the open-vocabulary detector"),
}


def _check_dependencies(required: dict) -> bool:
    import importlib.util

    missing = [(package, why) for module, (package, why) in required.items()
               if importlib.util.find_spec(module) is None]
    if not missing:
        return True

    print("PolliVision needs a few packages that are not installed yet:\n")
    for package, why in missing:
        print(f"  {package:<20} {why}")
    print("\nInstall them with:\n")
    print(f"  {sys.executable} -m pip install -r "
          f"{(PACKAGE_ROOT / 'requirements.txt').relative_to(ROOT) if PACKAGE_ROOT != ROOT else 'requirements.txt'}")
    print("\nIn VS Code: Terminal > Run Task > 'PolliVision: install'.")
    return False


def _check_gui() -> None:
    """Warn about the one dependency mistake that only shows up at display time."""
    try:
        from pollivision.io.webcam import gui_available, gui_hint
    except Exception:  # noqa: BLE001 - dependency check above reports the real cause
        return
    if not gui_available():
        print("\nNote: " + gui_hint() + "\n")


def main() -> int:
    argv = sys.argv[1:]
    if not _check_dependencies(CAPTURE_DEPS):
        return 2

    from pollivision.cli import main as cli_main

    if any(arg in {"-h", "--help"} for arg in argv):
        return cli_main(["webcam", "--help"])
    if "--list-cameras" in argv:
        return cli_main(["webcam", "--list-cameras"])

    if not _check_dependencies(MODEL_DEPS):
        return 2
    _check_gui()
    return cli_main(["webcam", *argv])


if __name__ == "__main__":
    sys.exit(main())
