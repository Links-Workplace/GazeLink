GAZELINK — אפיון טכני מעודכן

0. היררכיית המסמכים

ROADMAP.md — מקור האמת למטרות המוצר, Scope, Demo ו־Definition of Done של כל Milestone.

TECHNICAL_SPEC.md — מקור האמת לארכיטקטורה, Contracts, Safety, Privacy והחלטות טכניות.

TASKS.md — תוכנית הביצוע והסטטוס בפועל.

במקרה של סתירה אין להמציא Product Decision חדש. מציגים את הפער ומיישרים קו מול ה־Roadmap.

1. מטרת המערכת

GAZELINK היא אפליקציית Desktop מקומית שמקבלת וידאו ממצלמה ומאפשרת שליטה ב־Windows באמצעות:

Gaze — להבין איפה המשתמש מסתכל.

Eye Gestures — להבין מתי המשתמש רוצה לבצע פעולה מכוונת.

החיבור ביניהם מאפשר:

Look → Screen X/Y → Intentional Blink/Wink/Dwell → Action

Blink/Wink/Dwell הם חלק מתוכנן מראש מהמוצר ולא תוספת מאוחרת.

2. מצב טכני נוכחי — 2026-09-01

Foundation

בוצע מקומית:

Repository ושלד Python.

Domain contracts.

Configuration.

Logging / Metrics / Privacy guards.

Test infrastructure.

CI workflow ו־quality gates מקומיים.

נותר:

הרצת CI מרוחקת לאחר Commit/Remote.

אישור פורמלי של דגם Webcam כ־Reference device.

M1 — Vision

בנוי:

OpenCVCameraSource

פתיחת מצלמה בפועל ב־1280×720@30.

MediaPipe Face Landmarker.

Face/Eye/Iris/Eyelid extraction.

EyeFeatures עם:

iris_center בקואורדינטות Frame.

iris_in_eye בקואורדינטות יחסיות לעין.

חוזה `iris_in_eye` ב־Feature schema 2: ‏X נמדד לאורך ציר פינות העין; ‏Y נמדד כהיסט הניצב מאותו ציר, מחולק ברוחב העין ומוזז כך שקו הפינות הוא 0.5. העפעפיים משמשים ל־openness ולכיוון הציר האנכי בלבד, לא כנקודות הייחוס של Y, כדי שתנועת עפעפיים יחד עם הקשתית לא תבטל מבט אנכי. שינוי ההגדרה פוסל במפורש Dataset/Model/Correction מסכמה 1 במקום לפרש אותם בשקט לפי החוזה החדש.

openness.

Head Pose.

ConfidencePolicy.

Lost / Recovered.

VisionRuntime.

Live Debug Overlay ב־PySide6.

Terminal diagnostics.

Metrics, cleanup ו־privacy protections.

מדידות חיות שנאספו:

FPS אפקטיבי: בערך 28.4–28.6.

Latency ממוצע: בערך 10–13ms.

P95 שנמדד: בערך 13–17ms.

נותר לפני סגירת M1:

Live visual alignment על מסך אמיתי.

אימות חוזר ל־--debug-overlay לאחר תיקון התקיעה.

10-minute soak test עם CPU/Memory.

Multiple faces.

Intentional single-eye closure scenario.

M2 — Gaze

בנוי:

CalibrationSession.

9-point target model.

Guided Calibration UI.

Minimum sample gating.

Auto progression.

Retry / Restart / Cancel logic.

Calibration logs.

Completion אמיתי של 9 נקודות מול מצלמה ומשתמש.

פער קריטי נוכחי:

הכיול כרגע יודע לספור Samples לכל נקודה, אבל עדיין אינו שומר Dataset מלא של תכונות העין/הראש מול נקודת המסך.

לכן עדיין אין:

Training dataset.

Calibration Mapping model.

GazeEstimator שמחזיר X/Y.

Gaze filtering.

Accuracy validation.

Gaze point demo.

זהו הצעד הטכני הבא המרכזי.

3. Stack מאושר

רכיב

טכנולוגיה

מצב

Core

CPython 3.11.x

מאושר

Camera

OpenCV 4.11 מאחורי CameraSource

עובד בפועל

Face/Eye/Iris

MediaPipe 0.10.x מאחורי Adapter

עובד בפועל

Math

NumPy

מאושר

Desktop UI

PySide6 / Qt

עובד; נדרש QA על Display אמיתי

Windows Input

Win32 SendInput מאחורי Adapter

M3

Local Storage

SQLite / versioned file store

החלטה ב־M4

Local API

FastAPI/ASGI או מקביל

החלטה ב־M5

Packaging

TBD לפי CV/Qt/Windows

החלטה ב־M6

אין צורך ב־YOLO כל עוד MediaPipe מספק באופן יציב את ה־Face/Eye/Iris landmarks הדרושים.

4. ארכיטקטורת על

Webcam
  ↓
CameraSource
  ↓
VisionEngine
  ↓
FeatureExtractor
  ↓
ConfidencePolicy
  ↓
Calibration / GazeEstimator
  ↓
GazeFilter
  ↓
InteractionEngine
  ↓
SafetyController
  ↓
CursorController
  ↓
WindowsInputAdapter

במקביל:

VisionObservation
  ↓
Eye Openness
  ↓
Blink/Wink Detector
  ↓
Gesture State Machine
  ↓
Interaction Event

UI, Keyboard, WebSocket/API ו־SDK צורכים את אותם Domain contracts ולא בונים Pipeline חלופי.

5. Contracts מרכזיים

FramePacket

מכיל:

frame id

monotonic capture timestamp

width / height

pixel format

in-memory image buffer

ה־image אינו נשמר או מסוריאלז כברירת מחדל.

VisionObservation

מכיל:

tracking state

face geometry

left/right eye features

iris

openness

head pose

confidence

reason codes

CalibrationSample — נדרש ב־M2

יש להוסיף Contract מפורש עבור Sample אמיתי:

source frame id

captured/observed timestamp

target screen position

left iris-in-eye

right iris-in-eye

eye openness/visibility

head yaw/pitch/roll

face geometry/scale לפי הצורך

quality/confidence

accepted/rejected reason

אין לשמור Raw frame בתוך ה־Calibration dataset.

GazeSample

יכלול:

raw normalized X/Y

filtered normalized X/Y

screen X/Y

screen id/geometry

confidence

valid-for-control

reason codes

InteractionEvent

יכלול:

MOVE

LEFT_CLICK

RIGHT_CLICK

DOUBLE_CLICK

DWELL_SELECT

SCROLL

DRAG_START / DRAG_END

PAUSE / RESUME

6. M1 — Vision

אחריות

לייצר VisionObservation אמין ממצלמה חיה.

כלל קריטי

M1 אינו אמור לדעת איפה המשתמש מסתכל על המסך. הוא רק מספק את חומרי הגלם.

Definition of Done טכני

Camera עובדת ומשתחררת בבטחה.

Face/Eyes/Iris/Eyelids מופקים בפועל.

Eye openness עובד.

Head Pose עובד.

Lost/Recovered עובד.

Overlay חי ומיושר נכון.

Session ממושך אינו דולף משאבים.

אין stale observations.

7. M2 — Gaze

אחריות

ללמוד את הקשר בין:

Eye/Iris/Head features → Known screen target

ולהפעיל את המודל על כל Frame חדש כדי לקבל:

Screen X/Y

7.1 שלב A — איסוף כיול

כבר קיים Guided 9-point calibration.

השלב הבא הוא להרחיב אותו כך שכל Sample שמתקבל ישמור CalibrationSample אמיתי ולא רק Count.

7.2 שלב B — Quality / Outlier

יש לסנן:

Tracking לא תקין.

Eye visibility לא מספקת.

Head Pose מוחלט חריג (סף קשיח, ייעודי ונדיב יותר לכיול מאשר ל-Live control — ראו M2-02 ב-TASKS.md).

Sample ישן.

Outliers סטטיסטיים בתוך אותה נקודת Target.

כל rejection מקבל Reason code.

הכרעת Outlier נעשית על חלון רציף ולא על Baseline שנבנה מהדגימות הראשונות: Observations שעברו את ה־Gates נשארים provisional עד שנמצא תת־חלון רציף בגודל המינימום שעומד ב־P95 של תנועת שתי העיניים ושל תנועת Head Pose קצרת־טווח. רק החלון שנבחר מתקבל; שאר המועמדים מקבלים `target_outlier`. אם אין חלון יציב, אותו Target חוזר ל־Stabilization אוטומטית. אין בכך Gate נגד תנוחת ראש שונה לאחר שהמשתמש התייצב על Target אחר.

הערה — Head Pose יחסי ל-Session אינו Gate: הניסוח הקודם כלל גם סינון לפי Head Pose חריג ביחס ל-Session (סטייה יחסית לשאר הכיול, לא סף מוחלט). הוחלט במפורש (M2-02 Stage B) שלא לסנן לפי זה: תנוחת הראש היא Feature שמודל ה-Gaze ב-M2-03 אמור ללמוד ממנה, ולא רעש שיש להסיר, וסינון כזה היה מקטין את מרחב האימון ומגדיל Over-rejection. הסטייה היחסית נחשפת כ-Metric תיאורי בלבד (`CalibrationSessionResult.head_pose_spread()`), ואינה משפיעה על accepted/rejected.

7.3 שלב C — Mapping

יש להשוות לפחות:

Baseline פשוט.

מודל מתקדם אחד.

המטרה אינה לבחור מודל "מרשים", אלא מודל שמספק Accuracy טוב עם Latency נמוך ויציבות.

7.4 שלב D — GazeEstimator

מקבל VisionObservation + CalibrationModel ומחזיר GazeSample.

7.5 שלב E — Filter

יש למדוד Jitter מול Added latency.

בחירה בין EMA / One Euro / Kalman נעשית לפי Benchmark.

7.6 שלב F — Validation

Validation חייב להשתמש בנקודות או Samples שלא שימשו ישירות לאימון.

Preflight חמשת הכיוונים של M2 הוא כלי אבחוני לפני Calibration, לא Gate של Control ולא תחליף ל־Validation. הפרדה אופקית דורשת אות מינימלי בכל עין. הפרדה אנכית נבחנת דו־עינית: Combined signal מינימלי, רצפת אות בכל עין והסכמה בסימן; כיוון הפוך או חוסר הסכמה נשארים כשל מפורש. הספים הם Placeholders מדודים ומתועדים ב־`TASKS.md`, נשמרים ב־Artifact ומוצגים בדוח. אין להעתיק את דרישת Head Pose הקבוע של ה־Preflight ל־Calibration, שבו Head Pose הוא Feature חוקי של המודל.

מדדים:

median error

P95 error

region/corner error

jitter

latency

M2 Demo

Guided calibration → validation → נקודה שעוקבת אחרי המבט.

8. M3 — Control

עיקרון

Gaze קובע איפה.

Eye Gesture קובע מתי לבצע פעולה.

8.1 Blink/Wink

ה־openness מ־M1 הוא חומר גלם בלבד.

נדרש מנוע מחוות שמבדיל:

Natural blink

Left wink

Right wink

Intentional hold

Double gesture

Long close / Pause

8.2 ברירת מחדל מוצרית לפי האפיון

המיפוי הסופי ייקבע ב־D-04/D-05, אך כיוון המוצר הוא:

Left wink → Left click

Right wink → Right click

Sequence → Double click

Long intentional gesture → Drag / special action

Long both-eyes close → Pause / main menu

Natural blink → Ignore

Dwell → Alternative selection mode

8.3 Safety

אין OS action כאשר:

Control לא הופעל במפורש.

Calibration לא תקף.

Tracking lost.

Confidence נמוך.

Pause/Error פעיל.

Event ישן או כפול.

כל Mouse-down חייב מסלול Release בכל Error/Shutdown.

9. M4 — Usability

כולל:

Smart snapping.

Zoom lens.

Scroll zones.

Accessible feedback.

On-screen keyboard.

Hebrew RTL + English.

Word prediction.

Profiles.

Personal thresholds.

שמירת Calibration/Settings.

Therapist settings בסיסיים יכולים להתחיל ב־M4; הרשאות והקשחה מלאות ב־M6.

10. M5 — Platform

GAZELINK חושף את אותם Domain events דרך:

Local WebSocket.

REST.

SDK.

אין חשיפת Raw frames/face landmarks ב־Public API.

Integrations:

Browser.

PANDY.

LAXI.

PANDY/LAXI נשארים חסומים עד קבלת חוזים טכניים אמיתיים.

11. M6 — Product

כולל:

Accuracy hardening.

Calibration refinement.

Robustness.

Installer.

Setup Wizard.

Environment Check.

Multiple Profiles.

Therapist Console.

Encryption.

Safe fail / crash recovery.

Hardware matrix.

Pilot.

Go / No-Go.

IR/macOS/Android הם הרחבות לאחר יציבות הליבה אלא אם מתקבלת החלטת מוצר אחרת.

12. Safety / Privacy / Accessibility

Safety

OS input כבוי כברירת מחדל.

Fail closed.

Pause/Emergency תמיד זמינים כש־Control פעיל.

Automated tests לעולם אינם משתמשים ב־Real Windows input.

Privacy

וידאו מעובד מקומית.

אין שמירת Frames.

אין Raw biometric landmarks בלוגים.

Calibration/Profile שומרים רק את הנדרש לתפעול.

Accessibility

ליבה ניתנת להפעלה ללא ידיים.

עברית RTL + אנגלית.

אין הסתמכות על צבע בלבד.

Target size / Dwell / sensitivity ניתנים להתאמה.

13. מדדי איכות

יעדי המוצר הסופיים נשמרים באפיון, אך בזמן הפיתוח כל טענה חייבת להיות מבוססת מדידה.

נמדד כבר ב־M1:

~28.4–28.6 FPS.

~10–13ms average Vision latency.

נדרש למדוד בהמשך:

Gaze accuracy.

Jitter.

Control latency.

False activation.

Long-session stability.

Recovery.

Resource usage.

14. Source-of-truth rule ל־Claude Code

לפני כל משימה:

קרא את ה־Milestone הרלוונטי ב־ROADMAP.md.

קרא את החלק הטכני המקביל במסמך זה.

קרא את המשימה ב־TASKS.md.

הסבר לעצמך איזו יכולת מוצרית המשימה מקדמת.

אין להציג Blink/Wink/Dwell, Calibration, Gaze mapping או Mouse actions כ"רעיונות חדשים של המשתמש" — אלה חלק מהמערכת המתוכננת מראש.

בכל סיום עבודה יש להפריד בין:

Implemented.

Validated automatically.

Validated with real camera.

Still pending manual/hardware QA.
