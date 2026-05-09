"""Match an unknown face encoding against stored student encodings."""

from __future__ import annotations

from typing import Any

import face_recognition
import numpy as np


def match_face(
    unknown_encoding: np.ndarray,
    known_encodings_list: list[dict[str, Any]],
    threshold: float = 0.5,
) -> dict[str, Any] | None:
    """
    Compare one unknown encoding to all known encodings from load_encodings().

    Uses face_recognition.compare_faces and face_distance.
    The best candidate must have face_distance <= threshold (same meaning as compare_faces tolerance).

    Returns {name, roll_number, confidence} with confidence = 1 - distance, or None if no match.
    """
    if not known_encodings_list:
        return None

    unknown = np.asarray(unknown_encoding, dtype=np.float64)
    paired: list[tuple[dict[str, Any], np.ndarray]] = []
    for entry in known_encodings_list:
        enc = entry.get("encoding")
        if enc is None:
            continue
        paired.append((entry, np.asarray(enc, dtype=np.float64)))

    if not paired:
        return None

    known_arrays = [p[1] for p in paired]
    distances = face_recognition.face_distance(known_arrays, unknown)
    matches = face_recognition.compare_faces(known_arrays, unknown, tolerance=threshold)

    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])

    if best_distance > threshold or not matches[best_index]:
        return None

    entry = paired[best_index][0]
    confidence = max(0.0, min(1.0, 1.0 - best_distance))

    return {
        "name": entry.get("name", ""),
        "roll_number": entry.get("roll_number", ""),
        "confidence": confidence,
    }
