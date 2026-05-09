"""Live webcam recognition session."""

from __future__ import annotations

import datetime as dt

import cv2

from recognition.detector import detect_and_recognise
from recognition.encoder import load_encodings

WINDOW_TITLE = "Attendance System — Press Q to quit"


def _draw_status_panel(
    frame,
    x: int,
    y: int,
    faces_session_total: int,
    marked_count: int,
    now: dt.datetime,
) -> None:
    """Semi-opaque panel top-left with session stats and clock."""
    lines = [
        f"Total faces detected this session: {faces_session_total}",
        f"Total attendance marked this session: {marked_count}",
        now.strftime("%Y-%m-%d %H:%M:%S"),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    pad = 8
    line_h = 22

    max_w = 0
    total_h = pad * 2 + line_h * len(lines)
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        max_w = max(max_w, tw)

    x2 = x + max_w + pad * 2
    y2 = y + total_h

    roi = frame[y:y2, x:x2]
    if roi.size == 0:
        return
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x, y2 - y), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, roi, 0.35, 0, roi)

    yy = y + pad + line_h - 4
    for line in lines:
        cv2.putText(frame, line, (x + pad, yy), font, scale, (240, 240, 240), thickness, cv2.LINE_AA)
        yy += line_h


def _draw_text_lines(
    frame,
    origin_x: int,
    origin_y: int,
    items: list[tuple[str, tuple[int, int, int]]],
    font_scale: float = 0.48,
    line_step: int = 18,
) -> None:
    """Draw multiple coloured text lines (BGR colours)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = origin_y
    for text, colour in items:
        cv2.putText(frame, text, (origin_x, y), font, font_scale, colour, 1, cv2.LINE_AA)
        y += line_step


def run_recognition_session(
    camera_index: int = 0,
    match_threshold: float = 0.5,
) -> list[dict[str, str | float]]:
    """
    Open webcam, run continuous face detection/recognition.

    Press Q to quit. Returns one entry per student marked at least once (first hit).
    """
    known = load_encodings()
    if not known:
        print("Error: no face encodings loaded. Run training (tools/train.py) first.")
        return []

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: could not open camera index {camera_index}")
        return []

    # Session state (in-memory only; no database writes here).
    already_marked: set[str] = set()
    marked_details: dict[str, dict[str, str | float]] = {}
    session_faces_total = 0
    session_matches: list[dict[str, str | float]] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: failed to read frame from camera")
                break

            display = frame.copy()
            now = dt.datetime.now()

            try:
                detections = detect_and_recognise(frame, known, match_threshold=match_threshold)
            except Exception as error:
                print(f"Detection error: {error}")
                detections = []

            session_faces_total += len(detections)

            _draw_status_panel(display, 10, 10, session_faces_total, len(already_marked), now)

            for det in detections:
                top, right, bottom, left = det["face_location"]
                roll = det.get("roll_number")
                name = det.get("name")
                conf = det.get("confidence")

                pad = 6
                inner_x = left + pad
                inner_y = top + pad + 14

                if roll is not None and conf is not None:
                    roll_str = str(roll)
                    full_name = str(name)
                    pct = float(conf) * 100.0

                    cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)

                    lines_to_draw: list[tuple[str, tuple[int, int, int]]] = [
                        (full_name, (255, 255, 255)),
                        (f"Match: {pct:.0f}%", (255, 255, 255)),
                    ]

                    if roll_str not in already_marked:
                        already_marked.add(roll_str)
                        marked_details[roll_str] = {"name": full_name, "confidence": float(conf)}
                        session_matches.append(
                            {"name": full_name, "roll_number": roll_str, "confidence": float(conf)}
                        )
                        lines_to_draw.append(("Attendance Marked ✓", (0, 255, 0)))
                    else:
                        lines_to_draw.append(("Already Marked", (0, 255, 255)))

                    _draw_text_lines(display, inner_x, inner_y, lines_to_draw)
                else:
                    cv2.rectangle(display, (left, top), (right, bottom), (0, 0, 255), 2)
                    _draw_text_lines(
                        display,
                        inner_x,
                        inner_y,
                        [
                            ("Unknown Person", (0, 0, 255)),
                            ("Not Marked", (0, 0, 255)),
                        ],
                    )

            cv2.imshow(WINDOW_TITLE, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    ended = dt.datetime.now()
    print("\n===== SESSION SUMMARY =====")
    # Stable order: numeric rolls sorted numerically, else lexicographic.
    def sort_key(r: str):
        return (0, int(r)) if r.isdigit() else (1, r)

    for roll in sorted(marked_details.keys(), key=sort_key):
        info = marked_details[roll]
        nm = str(info["name"])
        cf = float(info["confidence"]) * 100.0
        print(f"✓ {nm} ({roll}) — {cf:.0f}% match")

    print(f"Total marked: {len(marked_details)} students")
    print(f"Session ended: {ended.strftime('%H:%M:%S')}")

    return session_matches
