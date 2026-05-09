"""Live webcam recognition session."""

from __future__ import annotations

import cv2

from recognition.detector import detect_and_recognise
from recognition.encoder import load_encodings


def run_recognition_session(
    camera_index: int = 0,
    match_threshold: float = 0.5,
) -> list[dict[str, str | float]]:
    """
    Open webcam, run continuous face detection/recognition, return deduplicated matches.

    Press Q to quit. Matched students are deduplicated by roll_number in the returned list.
    """
    known = load_encodings()
    if not known:
        print("Error: no face encodings loaded. Run training (tools/train.py) first.")
        return []

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: could not open camera index {camera_index}")
        return []

    window_title = "Attendance - Press Q to quit"
    frame_count = 0
    seen_rolls: set[str] = set()
    session_matches: list[dict[str, str | float]] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: failed to read frame from camera")
                break

            frame_count += 1
            display = frame.copy()
            h, w = display.shape[:2]

            try:
                detections = detect_and_recognise(frame, known, match_threshold=match_threshold)
            except Exception as error:
                print(f"Detection error: {error}")
                detections = []

            face_count = len(detections)

            for det in detections:
                top, right, bottom, left = det["face_location"]
                roll = det.get("roll_number")
                name = det.get("name")
                conf = det.get("confidence")

                if roll is not None and conf is not None:
                    color = (0, 255, 0)
                    pct = conf * 100.0
                    label = f"{name} ({roll}) {pct:.1f}%"
                    if str(roll) not in seen_rolls:
                        seen_rolls.add(str(roll))
                        session_matches.append(
                            {"name": str(name), "roll_number": str(roll), "confidence": float(conf)}
                        )
                else:
                    color = (0, 0, 255)
                    label = "Unknown"

                cv2.rectangle(display, (left, top), (right, bottom), color, 2)
                label_y = max(top - 10, 20)
                cv2.putText(
                    display,
                    label,
                    (left, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            status = f"Frame: {frame_count} | Faces: {face_count}"
            cv2.putText(
                display,
                status,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_title, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return session_matches
