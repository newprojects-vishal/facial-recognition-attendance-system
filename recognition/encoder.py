"""Build and load face encodings from student photo files on disk."""

from __future__ import annotations

import pickle
from pathlib import Path

import face_recognition
import numpy as np

# Add photos manually here (see README naming rules), then run train / build_encodings().
DEFAULT_PHOTOS_DIR = Path("data/student_photos")
ENCODINGS_PATH = Path("data/encodings.pkl")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_photo_filename(stem: str) -> tuple[str, str]:
    """
    Parse rollnumber_firstname_lastname... from the file stem.

    Split on "_"; first token is roll number; remaining tokens are name words
    after dropping segments where str.isdigit() is true (variant suffixes).

    The display name is joined with spaces and passed through .title().

    Examples:
      101_Vishal_7 -> roll 101, name "Vishal"
      101_Vishal_Wadekar -> roll 101, name "Vishal Wadekar"
      101_Vishal_Wadekar_2 -> roll 101, name "Vishal Wadekar"
      101_vishal_3 -> roll 101, name "Vishal"
    """
    parts = stem.split("_")
    if len(parts) < 2:
        cleaned = stem.strip()
        return cleaned, cleaned.title() if cleaned else cleaned

    roll_number = parts[0].strip()
    name_segments = [p for p in parts[1:] if p and not p.isdigit()]
    joined = " ".join(name_segments).strip()
    name = joined.title() if joined else roll_number
    return roll_number, name


def build_encodings(photos_dir: str | Path = "data/student_photos") -> list[dict]:
    """
    Scan photos_dir for images, detect one face per image, encode with face_recognition.

    Filenames should follow rollnumber_firstname_lastname.jpg (e.g. 101_Vishal_Wadekar.jpg).

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
