"""Optional backend hooks — the main app marks attendance on screen only."""

from config.settings import SUPABASE_KEY, SUPABASE_URL


def test_connection() -> None:
    """
    Lightweight check for optional credentials in `.env`.

    This project does not require a database for the camera attendance flow.
    """
    url_set = bool(SUPABASE_URL and SUPABASE_URL.strip())
    key_set = bool(SUPABASE_KEY and SUPABASE_KEY.strip())

    if url_set and key_set:
        print("Optional backend credentials detected (unused in this build).")
    elif url_set or key_set:
        print("Warning: partial backend configuration (SUPABASE_URL / SUPABASE_KEY mismatch).")
    else:
        print("No backend database configured — attendance is shown on camera only.")
