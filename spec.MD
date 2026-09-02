GAZELINK — Roadmap מוצר מעודכן

המטרה הכללית

GAZELINK היא מערכת שמאפשרת לאדם לשלוט במחשב באמצעות העיניים בלבד.

המבט קובע איפה הסמן נמצא, ומחוות עיניים מכוונות משמשות לביצוע פעולות כמו קליק, בחירה, Pause ופעולות נוספות.

ה־End Goal הוא לא רק Eye Tracking, אלא לאפשר למשתמש שאינו מסוגל להשתמש בידיים להפעיל מחשב באופן עצמאי באמצעות העיניים.

תמונת מצב נוכחית

Milestone

מצב

M1 — Vision

🟡 כמעט הושלם; הליבה בנויה, נשאר QA פיזי וסגירת Demo

M2 — Gaze

🟡 בתהליך; איסוף כיול 9 נקודות עובד, עדיין אין מיפוי מבט ל־X/Y

M3 — Control

⬜ טרם התחיל

M4 — Usability

⬜ טרם התחיל

M5 — Platform

⬜ טרם התחיל

M6 — Product

⬜ טרם התחיל

חשוב: Blink/Wink/Dwell אינם רעיונות חדשים או הרחבות אופציונליות. הם חלק מהותי מהמערכת המתוכננת מראש ונכנסים ב־M3.

MILESTONE 1 — Vision: המערכת רואה ומבינה את העיניים

מטרה

לבנות את שכבת הראייה הבסיסית של GAZELINK.

המערכת צריכה לקבל וידאו מהמצלמה ולהבין בזמן אמת את מצב הפנים והעיניים.

Scope

חיבור למצלמת Webcam

קליטת וידאו בזמן אמת

זיהוי פנים

זיהוי עין ימין ועין שמאל

זיהוי Iris / אישון

זיהוי העפעפיים

חישוב Eye Openness

זיהוי תנוחת הראש

Confidence / Tracking state

FPS ו־Latency בסיסיים

Lost / Recovery

Live Debug Overlay

אימות אנושי מלא על מסך אמיתי שה־Overlay מיושר נכון

Session רציף של 10 דקות + CPU/Memory

תרחיש Multiple Faces

תרחיש עצימת עין מכוונת

אימות חוזר שה־Debug Overlay אינו נתקע בסביבת Windows האמיתית

מה כבר עובד בפועל

מצלמה נפתחת ונקראת בפועל ב־1280×720@30.

OpenCV + MediaPipe עובדים מול וידאו אמיתי.

פנים, שתי עיניים, Iris, openness ו־Head Pose מופקים מהמצלמה.

Runtime מחבר Camera → Vision → Confidence → Overlay.

קיימים Lost/Recovered ו־resource cleanup.

קיימת מערכת בדיקות ו־Quality gates.

Demo

וידאו חי מהמצלמה עם Overlay המציג פנים, עיניים, Iris, Eye Openness, Head Pose, Tracking state, FPS ו־Latency.

Definition of Done

GAZELINK מצליח לראות אותי ולהבין את העיניים שלי בצורה יציבה לאורך זמן.

MILESTONE 2 — Gaze: המערכת יודעת איפה אני מסתכלת

מטרה

להפוך את נתוני העיניים לנקודה אמיתית על המסך.

זהו ה־Gaze Engine המרכזי.

Scope

כיול

מודל Session של 9 נקודות

Guided Calibration UI

איסוף מינימום Samples לכל נקודה

מעבר אוטומטי בין 9 נקודות

Feedback למשתמש

Completion אמיתי מול מצלמה

Calibration session log

שמירת Feature Vector אמיתי לכל Sample

Sample quality מלא

Outlier rejection

Gaze Mapping

בניית Dataset: Eye/Iris + Head Pose → Screen target

בחירת מודל Mapping

CalibrationEngine שמאמן את המיפוי

GazeEstimator שמחזיר Screen X/Y

Confidence למיקום החזוי

טיפול במבט מחוץ לטווח

יציבות ואימות

Smoothing / Filtering

טיפול ב־Jitter

Validation על נקודות שלא שימשו לאימון

Accuracy metrics

Gaze point overlay

מצב נוכחי חשוב

מסך הכיול כבר עובד ומסוגל להשלים את כל 9 הנקודות מול משתמש אמיתי.

אבל כרגע הכיול עדיין לא מלמד את המערכת איפה המשתמש מסתכל: נאספו Counts של Samples, אך עדיין לא נשמר Dataset מלא של תכונות העין/הראש מול נקודות המסך ולא נבנה Mapping ל־X/Y.

זהו הפער המרכזי הבא.

Demo

לאחר כיול, מופיעה נקודה על המסך שעוקבת אחרי המקום שאליו המשתמש מסתכל.

Definition of Done

GAZELINK יודע איפה אני מסתכלת על המסך ומספק X/Y מדיד ויציב.

MILESTONE 3 — Control: העיניים מחליפות את העכבר

מטרה

לעבור מ־Eye Tracking לשליטה אמיתית במחשב.

שני מנגנונים שחייבים להתחבר

1. Gaze

איפה המשתמש מסתכל.

2. Eye Gestures

מתי המשתמש רוצה לבצע פעולה.

החיבור הוא:

Look at target → Gaze X/Y → Intentional eye gesture → Action

Scope

Cursor

SafetyController

Windows Input Adapter

חיבור Gaze ל־Cursor

הזזת הסמן לפי המבט

הקפאת הסמן ב־Lost/Low confidence

Simulation mode

Blink / Wink Detection

זיהוי מצמוץ טבעי

זיהוי קריצה שמאל

זיהוי קריצה ימין

מדידת משך עצימה

הבחנה בין Natural Blink ל־Intentional gesture

Cooldown / one action per gesture

Mouse Actions

Left Click

Right Click

Double Click

Dwell Click

Pause / Emergency

Drag & Drop

מה כבר קיים מ־M1

Eye Openness לכל עין

תשתית זמן Monotonic

Tracking/Lost state

Confidence infrastructure

אלה חומרי הגלם של מנגנון ה־Blink/Wink, אך מנוע המחוות עצמו עדיין לא נבנה.

Demo

בלי לגעת בעכבר:

להסתכל על Chrome

להזיז אליו את הסמן

לבצע קריצה/מחווה מכוונת

לפתוח אותו

ללחוץ על כפתור

לבחור Tab

Pause וחזרה בטוחה

Definition of Done

אני יכולה לשלוט בעכבר ובמחשב באמצעות העיניים בלבד.

זהו ה־POC המרכזי של GAZELINK.

MILESTONE 4 — Usability: שליטה שימושית במחשב

מטרה

להפוך את ה־POC למערכת שאפשר באמת להשתמש בה לאורך זמן.

Scope

דיוק וחוויית שימוש

Smart Snapping

Target Expansion

Zoom Lens

Gaze Cursor

Feedback ויזואלי וקולי

Scroll באמצעות מבט

Dwell מתקדם

רגישות אישית

הקלדה

On-Screen Keyboard

עברית RTL

אנגלית

Blink/Dwell לבחירה

Word Prediction

משפטים נפוצים

משתמשים

User Profiles

שמירת Calibration

Blink/Wink thresholds

Smoothing settings

Dwell / sensitivity

Profile selection

Therapist settings בסיסיים

Demo

Browser → Scroll → Text field → Type → Send, ללא שימוש בידיים.

Definition of Done

אני יכולה באמת להשתמש במחשב באמצעות GAZELINK.

זהו שלב ה־Alpha.

MILESTONE 5 — Platform: GAZELINK הופך למנוע קלט

מטרה

לאפשר לאפליקציות אחרות להשתמש ב־Eye Tracking וב־Eye Control של GAZELINK.

Scope

Interface

Local Event Bus

WebSocket

Gaze coordinates

Gaze Enter / Leave

Select events

Blink/Wink events

Confidence

Calibration status

API

REST API

Profiles

Calibration control

Status

Basic statistics

Integrations

SDK בסיסי

Browser integration

PANDY

LAXI

Demo

מבט על סמל → Blink/Dwell → Select → Speech בתוך אפליקציה חיצונית.

Definition of Done

אפליקציות אחרות יכולות להשתמש ב־GAZELINK כמנוע קלט באמצעות העיניים.

זהו שלב ה־Beta.

MILESTONE 6 — Product: V1 למשתמשים אמיתיים

מטרה

להפוך את המערכת למוצר יציב, בטוח, נגיש וניתן להתקנה.

Scope

Reliability

Gaze Accuracy hardening

Calibration refinement

False Click reduction

תאורה משתנה

משקפיים

תנועות ראש

רעד

Recovery

Performance / Latency optimization

Product

Installer

Auto Start

Setup Wizard

Environment Check

Guided Calibration כחלק מהמוצר

Settings

Multiple Profiles

Therapist Console

Privacy & Safety

Local video processing

אין שמירת וידאו

Profile encryption

Pause / Emergency

Safe Fail

User / Therapist permissions

Verification

Full test suite

Hardware matrix

Performance / Accuracy tests

משתמשים אמיתיים

Pilot

Go / No-Go

הרחבות לאחר יציבות הליבה

IR Eye Tracker

macOS

Android

Cloud profile sync לפי החלטת מוצר

Demo

Install → Setup → Calibration → Eye Control → Pause → Safe shutdown

Definition of Done

GAZELINK הוא מוצר עצמאי שמוכן לשימוש אצל משתמשי קצה.

Roadmap במבט אחד

M1 — Vision
המערכת מבינה את העיניים.

↓

M2 — Gaze
המערכת יודעת איפה אני מסתכלת.

↓

M3 — Control
העיניים מחליפות את העכבר.

↓

M4 — Usability
אני יכולה באמת להשתמש במחשב.

↓

M5 — Platform
אפליקציות אחרות משתמשות ב־GAZELINK.

↓

M6 — Product
GAZELINK מוכן למשתמשים אמיתיים.

עיקרון העבודה

Camera, MediaPipe, Head Pose, Smoothing, EAR, State Machines, APIs וכדומה אינם Milestones בפני עצמם.

אלה משימות טכניות בתוך Milestone מוצרי.

Milestone צריך לענות על השאלה:

איזו יכולת משמעותית חדשה יש למשתמש בסוף השלב שלא הייתה לו קודם?

לכל Milestone חייבים להיות:

מטרה ברורה.

Scope ברור.

Demo שאפשר לבצע.

Definition of Done חד־משמעי.

תוצאה שקל להסביר גם למי שאינו מפתח.