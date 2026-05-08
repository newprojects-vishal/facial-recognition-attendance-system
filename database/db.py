"""Supabase database client helpers."""

from supabase import Client, create_client

from config.settings import SUPABASE_KEY, SUPABASE_URL


_supabase_client: Client | None = None


def get_client() -> Client:
    """Return a configured Supabase client."""
    global _supabase_client

    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase credentials are missing. Check your .env file.")

        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

    return _supabase_client


def test_connection() -> None:
    """Test the Supabase connection by querying the students table."""
    try:
        client = get_client()
        client.table("students").select("*").limit(1).execute()
        print("Supabase connection successful.")
    except Exception as error:
        print(f"Supabase connection failed: {error}")
