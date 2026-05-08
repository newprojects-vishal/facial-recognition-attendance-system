"""Application entry point."""

from database.db import test_connection
from registration.register_student import register_student


def _print_menu() -> None:
    """Display the main application menu."""
    print("\nSelect an option:")
    print("1. Register new student")
    print("2. Start attendance")
    print("3. Exit")


def main() -> None:
    """Start the facial recognition attendance system menu."""
    print("Facial Recognition Attendance System - Starting...")
    test_connection()

    while True:
        _print_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            register_student()
        elif choice == "2":
            print("Start attendance feature coming soon.")
        elif choice == "3":
            print("Exiting Facial Recognition Attendance System.")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
