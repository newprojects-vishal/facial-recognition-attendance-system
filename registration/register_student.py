"""Student registration workflow."""

import json
from pathlib import Path

import cv2
import face_recognition

from database.db import get_client, insert_student


PHOTO_DIR = Path("data/student_photos")
STORAGE_BUCKET = "student-photos"


def _collect_student_details() -> dict:
    """Ask the operator for the student's basic registration details."""
    return {
        "name": input("Enter student name: ").strip(),
        "roll_number": input("Enter roll number: ").strip(),
        "class": input("Enter class: ").strip(),
        "division": input("Enter division: ").strip(),
        "email": input("Enter email: ").strip(),
        "phone": input("Enter phone: ").strip(),
    }


def _upload_photo_to_supabase(photo_path: Path, storage_path: str) -> str:
    """Upload a student photo to Supabase Storage and return its public URL."""
    try:
        client = get_client()
        with photo_path.open("rb") as photo_file:
            client.storage.from_(STORAGE_BUCKET).upload(
                path=storage_path,
                file=photo_file,
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )

        return client.storage.from_(STORAGE_BUCKET).get_public_url(storage_path)
    except Exception as error:
        print(f"Failed to upload photo to Supabase Storage: {error}")
        raise


def _save_frame(frame, roll_number: str) -> Path:
    """Save the captured frame locally under the student's roll number."""
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    photo_path = PHOTO_DIR / f"{roll_number}.jpg"
    cv2.imwrite(str(photo_path), frame)
    return photo_path


def register_student() -> None:
    """Register a student by capturing their face from the webcam."""
    student_details = _collect_student_details()
    name = student_details["name"]
    roll_number = student_details["roll_number"]

    if not name or not roll_number:
        print("Name and roll number are required.")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Unable to open webcam.")
        return

    window_title = "Register Student - Press S to capture, Q to quit"
    print("Press S to capture the student's photo or Q to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read from webcam.")
                break

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Registration cancelled.")
                break

            if key != ord("s"):
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if not face_locations:
                print("No face detected, try again")
                cv2.putText(
                    frame,
                    "No face detected, try again",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow(window_title, frame)
                cv2.waitKey(1000)
                continue

            top, right, bottom, left = face_locations[0]
            face_encodings = face_recognition.face_encodings(rgb_frame, [face_locations[0]])
            if not face_encodings:
                print("Unable to generate face encoding, try again.")
                continue

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow(window_title, frame)
            cv2.waitKey(500)

            face_encoding_json = json.dumps(face_encodings[0].tolist())
            photo_path = _save_frame(frame, roll_number)
            storage_path = f"{roll_number}.jpg"
            profile_photo_url = _upload_photo_to_supabase(photo_path, storage_path)

            student_data = {
                **student_details,
                "face_encoding": face_encoding_json,
                "profile_photo_url": profile_photo_url,
                "is_active": True,
            }
            inserted_student = insert_student(student_data)

            if inserted_student:
                print(f"Student {name} registered successfully!")
            else:
                print("Student registration failed while inserting database record.")
            break
    except Exception as error:
        print(f"Student registration failed: {error}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    register_student()
