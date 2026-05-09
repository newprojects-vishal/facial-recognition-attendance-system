# Facial Recognition Attendance System

A Python-based facial recognition system that detects student faces through the webcam, identifies them against trained encodings, and marks attendance **on screen** during the session.

## Tech stack

- Python  
- OpenCV (`opencv-python`)  
- [`face_recognition`](https://github.com/ageitgey/face_recognition) (uses **dlib** under the hood)  
- NumPy  
- python-dotenv  
- Pillow  

## How it works

1. You add student photos under `data/student_photos/` using the naming rules below.  
2. Run **Train / rebuild face encodings** to write `data/encodings.pkl`.  
3. Run **Start attendance session** — the camera opens, faces are detected, matched students show **Attendance marked ✓** (once per roll number per session).  

There is no dashboard and no database in the main flow.

## Setup

1. Clone the repo and enter the project folder:

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

   On Windows, `dlib` may require extra steps (for example Visual C++ build tools, or a prebuilt wheel such as `dlib-bin`); see the [`face_recognition`](https://github.com/ageitgey/face_recognition) installation notes if install fails.

4. Optional: copy `.env.example` to `.env` if you use `config/settings.py` for future options. It is **not** required for the camera attendance flow.

5. Add photos and train (see below), then run:

   ```bash
   python main.py
   ```

## Adding a student

1. Add an image file to **`data/student_photos/`**.  
2. Name it: **`rollnumber_firstname_lastname.jpg`** (underscores between parts). Examples:

   - `101_Vishal_Wadekar.jpg` → roll **101**, name **Vishal Wadekar**  
   - `101_Vishal.jpg` → roll **101**, name **Vishal**  

   Numeric-only suffix segments (e.g. `_2`, `_3`) are ignored when building the display name.

3. Re-run training: menu option **1**, or:

   ```bash
   python tools/train.py
   ```

## Project layout

```text
facial-recognition-attendance-system/
├── main.py
├── requirements.txt
├── README.md
├── .env.example
├── config/
│   └── settings.py
├── database/
│   └── db.py          # optional / placeholder checks only
├── recognition/       # detector, encoder, matcher, camera
├── tools/
│   └── train.py       # rebuild encodings.pkl
└── data/
    ├── student_photos/
    └── encodings.pkl  # created after training
```
