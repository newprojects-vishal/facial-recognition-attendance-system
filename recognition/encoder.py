"""Face encoding module."""

import json

import numpy as np

from database.db import get_all_students


def load_known_faces() -> tuple[list[np.ndarray], list[str]]:
    """Load active student face encodings and IDs from the database."""
    known_encodings: list[np.ndarray] = []
    known_ids: list[str] = []

    try:
        students = get_all_students()
        for student in students:
            face_encoding = student.get("face_encoding")
            student_id = student.get("id")

            if not face_encoding or not student_id:
                continue

            try:
                encoding_array = np.array(json.loads(face_encoding), dtype=np.float64)
                known_encodings.append(encoding_array)
                known_ids.append(student_id)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                print(f"Skipping invalid face encoding for student {student_id}: {error}")

        return known_encodings, known_ids
    except Exception as error:
        print(f"Failed to load known faces: {error}")
        return [], []
