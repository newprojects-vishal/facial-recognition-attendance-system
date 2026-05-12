"""Detect faces in frames and recognise them against known encodings."""

from __future__ import annotations

from typing import Any

import cv2
import face_recognition
import numpy as np

from recognition.matcher import match_face


def detect_and_recognise_fast(
    frame_bgr: Any,
    known_encodings_list: list[dict[str, Any]],
    match_threshold: float = 0.5,
    scale: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Fast detection + recognition pipeline for live camera use.

    Detects faces on a downscaled frame (speed), but computes encodings on the
    full-size frame (dlib-bin compatibility — small images cause descriptor errors).

    Returns results with face_location in FULL frame pixel coordinates.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    inv_scale = int(1.0 / scale)

    small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
    small_rgb = np.ascontiguousarray(small[:, :, ::-1])

    face_locations_small = face_recognition.face_locations(small_rgb)
    if not face_locations_small:
        return []

    # Scale locations back to full resolution for encoding on the full image.
    face_locations_full = [
        (top * inv_scale, right * inv_scale, bottom * inv_scale, left * inv_scale)
        for (top, right, bottom, left) in face_locations_small
    ]

    full_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    face_encodings = face_recognition.face_encodings(full_rgb, face_locations_full)

    results: list[dict[str, Any]] = []
    for encoding, location in zip(face_encodings, face_locations_full):
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
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
