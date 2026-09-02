# GAZELINK

GAZELINK היא מערכת מקומית לשליטה בטוחה במחשב Windows באמצעות העיניים.

הפרויקט נמצא כרגע בשלב Foundation לקראת M1 — Vision. ה־Entry point הקיים הוא Smoke בטוח בלבד: הוא אינו פותח מצלמה ואינו מפעיל קלט של Windows.

## דרישות פיתוח

- Windows 11 x64 כמערכת הייחוס הראשונית.
- Python 3.11.x.
- PowerShell.
- Webcam תחובר ותיבחר לפני Hardware spike של M1.

## התקנה מקומית

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

`requirements.lock` נועד להתקנה שחזורית. `pyproject.toml` הוא מקור האמת לטווחי התלויות הישירות.

לפני בדיקות MediaPipe שאינן משתמשות במצלמה, מורידים את מודל ה־Face Landmarker המאומת:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
```

המודל נשמר תחת `.gazelink/models`, אינו נכנס ל־Git ומאומת מול SHA-256 נעול.

## Smoke בטוח

```powershell
.\.venv\Scripts\python.exe -m gazelink --smoke
```

התוצאה צריכה לאשר במפורש:

- `camera_enabled=false`
- `os_input_enabled=false`

גם הרצה ללא פרמטרים נשארת במצב בטוח ואינה מפעילה מצלמה או עכבר:

```powershell
.\.venv\Scripts\python.exe -m gazelink
```

## בדיקות ופיקוח איכות

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
```

להרצת כל השערים, כולל Build, בפקודה אחת:

```powershell
.\scripts\run_quality.ps1
```

בדיקות חומרה אינן חלק מפקודת ברירת המחדל ויתווספו תחת Marker מפורש. אסור לטעון Real Windows input adapter בבדיקות אוטומטיות.

## מסמכי מקור

- `spec.MD` — אפיון המוצר וה־Roadmap.
- `TECHNICAL_SPEC.md` — הארכיטקטורה והדרישות הטכניות.
- `TASKS.md` — תוכנית הביצוע והסטטוסים.
- `CLAUDE.md` — כללי עבודה וסוכנים.

## פרטיות ובטיחות

- וידאו מעובד מקומית בלבד.
- אין שמירת Frames כברירת מחדל.
- אין OS input לפני M3 וללא הפעלה מפורשת.
- במצב בדיקות משתמשים ב־Fake adapters בלבד.
