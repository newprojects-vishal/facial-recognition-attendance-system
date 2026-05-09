"""Build and load face encodings from student photo files on disk."""

from __future__ import annotations

import pickle
from pathlib import Path

import face_recognition
import numpy as np

# Default folder where registration saves captures (see registration/register_student.py).
DEFAULT_PHOTOS_DIR = Path("data/student_photos")
ENCODINGS_PATH = Path("data/encodings.pkl")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_photo_filename(stem: str) -> tuple[str, str]:
    """
    Expect rollnumber_studentname (e.g. 101_Vishal).
    Falls back to using the whole stem as roll number if there is no underscore.
    """
    if "_" in stem:
        roll_part, name_part = stem.split("_", 1)
        roll_number = roll_part.strip()
        name = name_part.replace("_", " ").strip() or roll_number
        return roll_number, name
    return stem.strip(), stem.strip()


def build_encodings(photos_dir: str | Path = "data/student_photos") -> list[dict]:
    """
    Scan photos_dir for images, detect one face per image, encode with face_recognition.

    Filenames should be rollnumber_studentname.jpg (e.g. 101_Vishal.jpg).

    Returns a list of dicts: {name, roll_number, encoding} where encoding is a numpy array.
    Also saves the list to data/encodings.pkl.
    """
    photos_path = Path(photos_dir)
    results: list[dict] = []

    if not photos_path.is_dir():
        print(f"Warning: photos directory not found: {photos_path.resolve()}")
        ENCODINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with ENCODINGS_PATH.open("wb") as handle:
                pickle.dump(results, handle)
        except OSError as error:
            print(f"Failed to write empty encodings file: {error}")
        return results

    image_files = sorted(
        p for p in photos_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"No image files found in {photos_path.resolve()}")

    for image_path in image_files:
        roll_number, name = _parse_photo_filename(image_path.stem)

        try:
            image = face_recognition.load_image_file(str(image_path))
        except Exception as error:
            print(f"Warning: could not load {image_path.name}: {error}")
            continue

        encodings = face_recognition.face_encodings(image)
        if not encodings:
            print(f"Warning: no face detected in {image_path.name} — skipped")
            continue

        if len(encodings) > 1:
            print(f"Warning: multiple faces in {image_path.name} — using the first")

        encoding_vector = np.asarray(encodings[0], dtype=np.float64)
        results.append({"name": name, "roll_number": roll_number, "encoding": encoding_vector})

    ENCODINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ENCODINGS_PATH.open("wb") as handle:
            pickle.dump(results, handle)
    except OSError as error:
        print(f"Failed to save encodings to {ENCODINGS_PATH}: {error}")
        raise

    print(f"Encoded {len(results)} face(s) successfully — saved to {ENCODINGS_PATH.resolve()}")
    return results


def load_encodings() -> list[dict]:
    """
    Load encodings from data/encodings.pkl.

    Each entry is {name, roll_number, encoding}.
    """
    if not ENCODINGS_PATH.is_file():
        print("No encodings found. Run build_encodings() first")
        return []

    try:
        with ENCODINGS_PATH.open("rb") as handle:
            data = pickle.load(handle)
    except (OSError, pickle.UnpicklingError) as error:
        print(f"Failed to load encodings: {error}")
        return []

    if not isinstance(data, list):
        print("Invalid encodings file format — expected a list")
        return []

    return data
