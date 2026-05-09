"""
Regenerate face encodings from student photos.

Run from the project root after adding or updating images in data/student_photos:
    python tools/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when executing this file directly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from recognition.encoder import build_encodings


def main() -> None:
    try:
        build_encodings()
        print("Training complete!")
    except Exception as error:
        print(f"Training failed: {error}")
        raise


if __name__ == "__main__":
    main()
