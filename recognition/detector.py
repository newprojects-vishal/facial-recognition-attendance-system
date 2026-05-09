"""Detect faces in frames and recognise them against known encodings."""

from __future__ import annotations

from typing import Any

import cv2
import face_recognition
import numpy as np

from recognition.matcher import match_face


def detect_and_recognise(
    frame: Any,
    known_encodings_list: list[dict[str, Any]],
    match_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Detect all faces in an OpenCV BGR frame and match each to known_encodings_list.

    Returns a list of dicts with keys:
      name, roll_number, confidence (None if unknown), face_location (top, right, bottom, left).
    """
    if frame is None or frame.size == 0:
        return []

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except cv2.error as error:
        print(f"Failed to convert frame to RGB: {error}")
        return []

    face_locations = face_recognition.face_locations(rgb)
    if not face_locations:
        return []

    face_encodings = face_recognition.face_encodings(rgb, face_locations)
    results: list[dict[str, Any]] = []

    for encoding, location in zip(face_encodings, face_locations):
        encoding_arr = np.asarray(encoding, dtype=np.float64)
        matched = match_face(encoding_arr, known_encodings_list, threshold=match_threshold)

        if matched:
            results.append(
                {
                    "name": matched["name"],
                    "roll_number": matched["roll_number"],
                    "confidence": matched["confidence"],
                    "face_location": location,
                }
            )
        else:
            results.append(
                {
                    "name": None,
                    "roll_number": None,
                    "confidence": None,
                    "face_location": location,
                }
            )

    return results
