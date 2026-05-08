"""Supabase database client helpers."""

from supabase import Client, create_client

from config.settings import SUPABASE_KEY, SUPABASE_URL


_supabase_client: Client | None = None


def get_client() -> Client:
    """Initialize and return the shared Supabase client."""
    global _supabase_client

    try:
        if _supabase_client is None:
            if not SUPABASE_URL or not SUPABASE_KEY:
                raise ValueError("Supabase credentials are missing. Check your .env file.")

            _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

        return _supabase_client
    except Exception as error:
        print(f"Failed to initialize Supabase client: {error}")
        raise


def test_connection() -> None:
    """Test the Supabase connection by querying the students table."""
    try:
        client = get_client()
        client.table("students").select("*").limit(1).execute()
        print("Supabase connection successful.")
    except Exception as error:
        print(f"Supabase connection failed: {error}")


def get_all_students() -> list[dict]:
    """Fetch all active students with their stored face encodings."""
    try:
        client = get_client()
        response = (
            client.table("students")
            .select("id, name, roll_number, face_encoding")
            .eq("is_active", True)
            .execute()
        )
        return response.data or []
    except Exception as error:
        print(f"Failed to fetch students: {error}")
        return []


def get_student_by_roll(roll_number: str) -> dict | None:
    """Fetch a single student record by roll number."""
    try:
        client = get_client()
        response = (
            client.table("students")
            .select("*")
            .eq("roll_number", roll_number)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None
    except Exception as error:
        print(f"Failed to fetch student with roll number {roll_number}: {error}")
        return None


def insert_student(data: dict) -> dict | None:
    """Insert a new student record and return the inserted row."""
    try:
        client = get_client()
        response = client.table("students").insert(data).execute()
        return response.data[0] if response.data else None
    except Exception as error:
        print(f"Failed to insert student: {error}")
        return None


def insert_attendance(
    student_id: str,
    date: str,
    time_in: str,
    status: str,
    confidence_score: float,
) -> dict | None:
    """Insert attendance or return today's existing record for the student."""
    try:
        client = get_client()
        existing_response = (
            client.table("attendance")
            .select("*")
            .eq("student_id", student_id)
            .eq("date", date)
            .limit(1)
            .execute()
        )

        if existing_response.data:
            return {
                "message": "Already marked today",
                "record": existing_response.data[0],
            }

        attendance_data = {
            "student_id": student_id,
            "date": date,
            "time_in": time_in,
            "status": status,
            "confidence_score": confidence_score,
        }
        response = client.table("attendance").insert(attendance_data).execute()
        return response.data[0] if response.data else None
    except Exception as error:
        print(f"Failed to insert attendance: {error}")
        return None


def get_attendance_by_date(date: str) -> list[dict]:
    """Fetch attendance for a date with student name and roll number."""
    try:
        client = get_client()
        response = (
            client.table("attendance")
            .select("*, students(name, roll_number)")
            .eq("date", date)
            .order("time_in")
            .execute()
        )
        return response.data or []
    except Exception as error:
        print(f"Failed to fetch attendance for {date}: {error}")
        return []


def get_student_attendance_history(student_id: str) -> list[dict]:
    """Fetch one student's attendance records ordered by newest date first."""
    try:
        client = get_client()
        response = (
            client.table("attendance")
            .select("*")
            .eq("student_id", student_id)
            .order("date", desc=True)
            .execute()
        )
        return response.data or []
    except Exception as error:
        print(f"Failed to fetch attendance history for student {student_id}: {error}")
        return []
