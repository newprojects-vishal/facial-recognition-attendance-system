# Facial Recognition Attendance System

A Python-based attendance system that uses facial recognition to identify students and records attendance data in Supabase.

## Tech Stack

- Python
- OpenCV
- face-recognition
- NumPy
- Supabase
- python-dotenv
- Pillow
- pandas
- openpyxl

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd facial-recognition-attendance-system
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy the environment template:

   ```bash
   copy .env.example .env
   ```

5. Fill in your Supabase credentials and attendance settings in `.env`.

6. Run the application:

   ```bash
   python main.py
   ```

## Folder Structure

```text
facial-recognition-attendance-system/
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── config/
│   └── settings.py
├── database/
│   └── db.py
├── recognition/
│   ├── __init__.py
│   ├── detector.py
│   ├── encoder.py
│   └── matcher.py
├── registration/
│   ├── __init__.py
│   └── register_student.py
├── attendance/
│   ├── __init__.py
│   ├── mark_attendance.py
│   └── rules.py
├── data/
│   └── student_photos/
│       └── .gitkeep
└── utils/
    ├── __init__.py
    └── helpers.py
```
