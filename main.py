"""Application entry point."""

from database.db import test_connection


def main() -> None:
    """Start the facial recognition attendance system."""
    print("Facial Recognition Attendance System - Starting...")
    test_connection()


if __name__ == "__main__":
    main()
