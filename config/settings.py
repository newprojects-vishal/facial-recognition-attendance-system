"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
ATTENDANCE_START_TIME = os.getenv("ATTENDANCE_START_TIME", "09:00")
ATTENDANCE_END_TIME = os.getenv("ATTENDANCE_END_TIME", "11:00")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
