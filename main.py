"""Application entry point — train encodings or run live attendance."""

from recognition.camera import run_recognition_session


def _print_menu() -> None:
    """Display the main application menu."""
    print("\nSelect an option:")
    print("1. Train / rebuild face encodings")
    print("2. Start attendance session")
    print("3. Exit")


def main() -> None:
    """Start the facial recognition attendance system."""
    print("Facial Recognition Attendance System")

    while True:
        _print_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            try:
                from recognition.encoder import build_encodings

                build_encodings()
                print("Training complete!")
            except Exception as error:
                print(f"Training error: {error}")
        elif choice == "2":
            try:
                run_recognition_session()
            except Exception as error:
                print(f"Attendance session error: {error}")
        elif choice == "3":
            print("Goodbye.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
