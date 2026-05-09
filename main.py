"""Application entry point."""

from database.db import test_connection
from recognition.camera import run_recognition_session
from registration.register_student import register_student


def _print_menu() -> None:
    """Display the main application menu."""
    print("\nSelect an option:")
    print("1. Register new student")
    print("2. Train / rebuild face encodings")
    print("3. Start attendance session")
    print("4. Exit")


def main() -> None:
    """Start the facial recognition attendance system menu."""
    print("Facial Recognition Attendance System - Starting...")
    try:
        test_connection()
    except Exception as error:
        print(f"Database connection check failed: {error}")

    while True:
        _print_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            try:
                register_student()
            except Exception as error:
                print(f"Registration error: {error}")
        elif choice == "2":
            try:
                from recognition.encoder import build_encodings

                build_encodings()
                print("Training complete!")
            except Exception as error:
                print(f"Training error: {error}")
        elif choice == "3":
            try:
                run_recognition_session()
            except Exception as error:
                print(f"Attendance session error: {error}")
        elif choice == "4":
            print("Exiting Facial Recognition Attendance System.")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
