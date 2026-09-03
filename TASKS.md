GAZELINK — TASKS מעודכן

0. מטרת המסמך

TASKS.md הוא תוכנית הביצוע של ROADMAP.md ו־TECHNICAL_SPEC.md.

המסמך החדש משתמש ב־Work Packages גדולים יותר במקום עשרות משימות קטנות, כדי שיהיה ברור מה היכולת שמתקדמת בכל שלב.

סטטוסים

✅ DONE

🟡 PARTIAL / IN PROGRESS

⬜ NOT STARTED

⛔ BLOCKED

FOUNDATION — בסיס רוחבי

F-01 — Technical Baseline

Status: 🟡 PARTIAL

בוצע

מחשב / Windows / CPU / RAM / GPU.

מסך / DPI / single-display scope.

Python / OpenCV / MediaPipe / NumPy / PySide6.

Webcam עובדת בפועל.

1280×720@30 נבדק.

Decision log.

נשאר

לאשר פורמלית איזה דגם Webcam הוא Reference device.

Legacy mapping

F-T01

F-02 — Project Foundation

Status: ✅ DONE locally

כולל

Repository structure.

Dependency management.

Domain contracts.

Configuration.

Errors.

Logging.

Metrics.

Privacy guards.

Fake clock/camera/input.

Unit/integration test structure.

Local quality gates.

נשאר מחוץ ל־DONE המקומי

Commit ראשון.

Remote.

GitHub Actions run אמיתי על checkout חדש.

Legacy mapping

F-T02 עד F-T06

M1 — Vision: המערכת רואה ומבינה את העיניים

M1-01 — Camera + Vision Core

Status: ✅ DONE

כולל

Camera open/read/close.

Resolution/FPS request.

Face detection.

Two eyes.

Iris.

Eyelids.

Live MediaPipe adapter.

Real camera validation.

ראיות

Webcam: 1280×720@30.

Real frames processed.

OpenCV + MediaPipe: GO.

Legacy mapping

M1-T01, M1-T02, M1-T04

M1-02 — Eye Features + Tracking State

Status: ✅ DONE

כולל

iris_center.

iris_in_eye.

Eye openness.

Head yaw/pitch/roll.

Confidence policy.

Lost/Recovered.

No stale observations.

הערה

Head-pose thresholds הם עדיין tuning values ולא ערכי מוצר סופיים.

Legacy mapping

M1-T05, M1-T06

M1-03 — Live Runtime + Diagnostics

Status: 🟡 PARTIAL

בוצע

Camera → Vision → Confidence → Overlay.

Latest-frame policy.

Debug Overlay.

PySide6 window.

Terminal diagnostics.

Safe cleanup.

Watchdog סביב camera mode request.

בדיקות Offscreen.

נשאר

להריץ שוב python -m gazelink --debug-overlay על מסך Windows אמיתי לאחר תיקון התקיעה.

לוודא ויזואלית שה־Overlay יושב נכון על הפנים/עיניים.

לוודא שהחלון נסגר נקי גם בתרחיש האמיתי.

Legacy mapping

M1-T03, M1-T07

M1-04 — M1 QA Gate

Status: 🟡 PARTIAL

בוצע

Automated regression.

Camera/vision short benchmarks.

FPS ~28.4–28.6.

Average latency ~10–13ms.

Lost/Recovery אמיתי.

165+ tests בסוויטה העדכנית.

Ruff / format / mypy נקיים.

נשאר

10-minute continuous session.

CPU/Memory measurement.

Multiple faces.

Intentional left/right eye close.

Final real-screen M1 demo.

Definition of Done

M1 נסגר כאשר אפשר להפעיל את המצלמה לאורך זמן ולראות באופן יציב פנים, עיניים, Iris, openness ו־Head Pose על המסך.

Legacy mapping

M1-T08, M1-T09

M2 — Gaze: המערכת יודעת איפה המשתמש מסתכל

M2-01 — Guided Calibration Collection

Status: 🟡 MOSTLY DONE

בוצע

9 calibration targets.

Stateful calibration session.

Minimum samples per target.

Auto progression.

Retry/Restart/Cancel logic.

Full-screen calibration UI.

Camera preview.

Low-quality feedback.

Calibration logs.

Real gaze-only completion מול משתמש.

Outcome: COMPLETE.

5/5 samples לכל 9 הנקודות בריצה אמיתית.

נשאר

Manual live check של Retry / Restart / Cancel.

Camera-loss בזמן Calibration.

Product review לפריסת 9 הנקודות על מסך ultrawide.

חשוב

השלב הזה מוכיח שאפשר לאסוף כיול, לא שהמערכת כבר יודעת X/Y.

Legacy mapping

M2-T01, M2-T02

M2-02 — Calibration Dataset + Sample Quality

Status: 🟡 PARTIAL — Stage A (real dataset storage) DONE ומאומת מול מצלמה אמיתית; Stage B (Quality policy/Outlier rejection) הקוד הושלם ואומת אוטומטית (236 tests + Scenario QA), אך טרם אומת מול מצלמה אמיתית וכל הספים החדשים הם Placeholders לא מכוילים

מטרה

להפוך כל Sample מהכיול לנתון שאפשר ללמוד ממנו.

בוצע (Stage A — Dataset storage)

נוסף CalibrationSample (src/gazelink/calibration.py) — Frozen dataclass מאומת, עם to_dict/from_dict.

נשמר target_index (איזה מ-9 היעדים).

נשמר left/right iris_in_eye.

נשמר left/right eye openness.

נשמר Head Pose (yaw/pitch/roll).

נשמר confidence.

לא נשמר Raw frame/image בשום שדה.

CalibrationSession.record_sample() משתנה מ-valid: bool ל-sample: CalibrationSample אמיתי; מאחסן accepted ו-rejected כאחד (לא רק Count).

CalibrationSessionResult.samples — ה-Dataset המלא, זמין רק ב-COMPLETE (אותה הגנה קיימת נגד Partial session).

retry_current_target() מנקה גם את ה-Samples השמורים של היעד הנוכחי, לא רק את ה-Count.

CalibrationUiController.ingest() בונה את ה-Sample האמיתי מה-VisionObservation שכבר יש לו (ללא צורך בעבודה נוספת ב-M1).

אימות: pytest 179 עברו (היה 165), ruff/format/mypy נקיים, python -m build הצליח.

אימות חי מול מצלמה אמיתית (Session אמיתי, לא Fixture): Outcome COMPLETE, sample_counts (3,3,3,3,3,3,3,3,3), 27 accepted + 11 rejected נשמרו. Sample מקובל לדוגמה: iris_in_eye אמיתי (לא (0.5,0.5) קבוע), openness 0.550/0.646, head pose אמיתי. Sample נדחה לדוגמה: כל שדות ה-Feature הם None (לא מזויפים), reason="tracking_not_ready". to_dict/from_dict round-trip תקין.

באג אמיתי שנמצא מקריאת Log אמיתי ששיתף המשתמש, ותוקן: 4 מתוך 5 Samples ב-Target אחד היו accepted=True עם כל שדה Feature = None (ריק לגמרי) — למרות ש-sample_counts הראה (5,5,5,...) "מושלם". הסיבה: VisionRuntime._display_observation() משאיר tracking_state=TRACKED במהלך חלון Debounce (כדי למנוע הבהוב UI), אבל מרוקן left_eye/right_eye/head_pose ל-None — בעוד ש-overall_confidence נשאר מה-Observation הגולמי (שיכול להיות 0.5). CalibrationUiController.ingest() בדק רק tracking_state ו-confidence, לא בדק אם ה-Features בפועל קיימים. תוקן: נוספה בדיקה מפורשת ל-left_eye/right_eye/head_pose לפני accept, עם reason="features_unavailable" חדש. בדיקת Regression נוספה (`test_tracked_observation_with_blanked_geometry_is_rejected_not_accepted`) שמשחזרת בדיוק את התרחיש. pytest 182 עברו (היה 179), ruff/format/mypy נקיים. אומת שוב חי מול מצלמה — 27/27 accepted עם Feature data אמיתי, 0 corrupted samples.

בוצע (Stage B — Quality policy) — הדרישות המקוריות והסטטוס שלהן

[✅] Accepted/rejected reason codes כ-Enum אמיתי (כרגע reason הוא str חופשי: "accepted"/"tracking_not_ready"/"low_confidence"/"features_unavailable" בלבד).
נוסף CalibrationReasonCode(StrEnum) ב-calibration.py עם 8 ערכים: ארבעת הערכים הקיימים נשמרו byte-identical (כדי שלוגים ו-Datasets שכבר נכתבו יישארו קריאים, ללא Migration) + stale_sample / eye_not_visible / head_pose_out_of_range / target_outlier. השדה CalibrationSample.reason נשאר מסוג str במכוון (to_dict/from_dict לא השתנו כלל), אך הערך מאומת מול הקבוצה הסגורה ב-__post_init__; העברת Member של ה-Enum מנורמלת ל-str רגיל.

[✅] Sample age checks (מעבר למה שכבר קיים ב-ConfidencePolicy).
Gate חדש ב-ingest() לפי max_sample_age_ms. חשוב: התגלה ותוקן פער אמיתי של בסיס-זמן — observation.observed_at_monotonic_ms נחתם ב-camera.py לפי perf_counter_ns()/1e6, בעוד confidence.monotonic_ms הוא time.monotonic()*1000 (שעון אחר עם Epoch אחר ב-Windows). שימוש ב-monotonic_ms כברירת מחדל היה עלול לסמן כל Sample תקין כ-stale ולשבור כיול אצל משתמש אמיתי, בעוד כל הבדיקות הדטרמיניסטיות (שמזריקות Fake clock) ממשיכות לעבור. לכן נוסף frame_monotonic_ms() ב-calibration_ui.py באותו בסיס-זמן של camera.py. נבדק חי: הפרש 0.000ms בין השעונים.
בנוסף: דחייה רק כאשר age > max, לעולם לא על age שלילי — Fail-open על תנאי בלתי-אפשרי, בניגוד מכוון ל-ConfidencePolicy שנכשל Closed (שם פעולת סמן שגויה מסוכנת; בכיול Over-rejection הוא הנזק הגדול יותר).

[✅] Eye visibility checks מפורטים (מעבר לבדיקת Tracking state הגסה שקיימת היום).
בדיקה לכל עין בנפרד: iris_in_eye is not None וגם openness >= min_eye_openness. זה תופס את המקרה שבו אובייקט EyeFeatures קיים (ולכן עובר את בדיקת features_unavailable) אך אינו שמיש בפועל — אמצע מצמוץ, או היעדר הקואורדינטה שנושאת את המבט.
EyeFeatures.confidence הוצא מהבדיקה במכוון: features.py מציב אותו מ-landmarks.detector_confidence, ו-vision.py מציב שם את הקבוע STRUCTURAL_CONFIDENCE=0.5. סינון לפיו היה משחזר באג קודם (סף על קבוע שלא ניתן לעבור). נוספה בדיקת Regression שמקבעת שאותו Confidence נמוך לבדו לעולם אינו דוחה Sample.

[✅] Head-pose quality checks יחסית ל-Session (לא רק הסף המוחלט הקיים מ-M1).
בוצע כ-Metric תיאורי בלבד ולא כ-Gate — ראו "סטייה מאושרת מ-TECHNICAL_SPEC 7.2" למטה. הסף היחיד שדוחה הוא סף מוחלט חדש ונפרד לכיול (CALIBRATION_HEAD_POSE_LIMITS: yaw 55°, pitch 40°, roll 40°), נדיב יותר במכוון מברירות המחדל של Live control (30/25/25) שנמדדו כדוחות זוויות לגיטימיות בטווח 36–50° בסשנים אמיתיים.

[✅] Outlier rejection בתוך כל Target — עדיין לא הוחלט על אלגוריתם; במכוון נדחה עד שיש Dataset אמיתי להסתכל עליו (יש עכשיו).
האלגוריתם שנבחר: Median רכיבי (median של ה-x-ים ו-median של ה-y-ים בנפרד, לכל עין בנפרד) על ה-Samples שכבר התקבלו ליעד הנוכחי, ומרחק אוקלידי של המועמד מול max_target_outlier_distance. Median ולא Mean במכוון — Sample רע בודד לא יכול לשלוט ב-Baseline. Bootstrap: שלושת ה-Samples הראשונים ליעד לעולם אינם נדחים כ-Outlier (אין עדיין Baseline). סיכון מוכר ומקובל: אם ה-Bootstrap עצמו רע — ה-Baseline רע; פעולת "Retry target" הקיימת היא מסלול המילוט המיועד, ואומת שהיא אכן מנקה Baseline "מורעל".

[✅] Metrics ל-accepted/rejected (Counters/Histogram, לא רק Log טקסט).
CalibrationSessionResult.reason_counts() ו-head_pose_spread() — שתיהן Derived מ-samples ולא שדות שמורים (כדי שלא ייווצר מקור אמת שני שעלול להיסחף). נוסף Section "--- Metrics ---" ל-format_calibration_log() בין "--- Final result ---" ל-"--- Dataset ---", עם מיון דטרמיניסטי לפי שם. במקרה ריק מוצג "none"/"unavailable" מפורש ולא אפסים מפוברקים (אפס היה מדווח בטעות "הראש לא זז כלל").

קבצים שהשתנו (Stage B)

src/gazelink/calibration.py — CalibrationReasonCode, אימות reason, current_target_accepted_samples, accepted_samples, reason_counts(), head_pose_spread().

src/gazelink/calibration_ui.py — frame_monotonic_ms(), CALIBRATION_HEAD_POSE_LIMITS, CalibrationQualitySettings, _median_point(), 4 Gates חדשים ב-ingest(), _record() מעביר Enum members.

src/gazelink/calibration_window.py — Section "--- Metrics ---" ב-format_calibration_log().

tests/test_calibration.py, tests/test_calibration_ui.py, tests/test_calibration_window.py.

אימות (Stage B)

pytest: 236 passed, 2 deselected (היה 182 לפני Stage B; +54 בדיקות). ruff check / ruff format --check / mypy (strict) נקיים. python -m build הצליח.

Scenario QA עצמאי של סוכן ה-Lead (מעבר ל-Unit tests): 26/26 עברו — Happy path, אין פנים, אובדן והתאוששות, כל עין בנפרד בשני האותות (openness ו-iris_in_eye), Confidence מבני נמוך שאינו דוחה, Sample ישן, גבול מדויק של גיל, חותמת "עתידית" (Fail-open), 45° מתקבל מול 60° נדחה, סחיפת ראש של 100° שאינה דוחה, Bootstrap ואז דחיית Outlier, Retry שמנקה Baseline, 5 Samples דחויים שאינם מקדמים יעד, גיאומטריה מרוקנת, Frame כפול, Cancel/Session חלקי שאינם מייצאים תוצאה, ותאימות בסיס-הזמן של השעון האמיתי.

תיקון שבוצע ב-QA של ה-Lead: שגיאת תיעוד ב-Docstring של CalibrationReasonCode (טווח הערכים ה"ישנים" נכתב כ-ACCEPTED..STALE_SAMPLE שהם חמישה, במקום ACCEPTED..FEATURES_UNAVAILABLE שהם ארבעה).

⚠ ספים חדשים — כולם Placeholders לא מכוילים (כמו DEFAULT_TARGET_EDGE_INSET, וכמו הערת M1-02 על Head-pose thresholds)

max_sample_age_ms = 500.0ms (Live control משתמש ב-250ms).

CALIBRATION_HEAD_POSE_LIMITS = yaw 55° / pitch 40° / roll 40°.

min_eye_openness = 0.15.

max_target_outlier_distance = 0.25 (במרחב iris_in_eye).

min_baseline_samples = 3.

⚠️ הרשומה לעיל מתעדת את מימוש Stage B המקורי. החל מ־Feature schema 2 היא הוחלפה בהחלטה המאושרת המתועדת תחת “Feature schema 2 + contiguous stability windows”: שני השדות הוסרו מהקוד, ואין עוד Bootstrap או Baseline מדגימות ראשונות.

אף אחד מהערכים לא כויל מול נתוני Accuracy אמיתיים. נדרש מעבר כיול מבוסס-מדידה לפני Release, ובפרט לאחר M2-03/M2-04 כשיהיה אפשר למדוד את השפעת כל סף על שגיאת הכיול בפועל. עקרון ההכרעה שנבחר: עדיף לדחות פחות מדי מאשר יותר מדי — משתמש שיכול להפעיל מחשב רק בעיניים אינו יכול "פשוט לנסות שוב עם עכבר".

✅ סתירה מול TECHNICAL_SPEC 7.2 — דווחה ותוקנה באישור המשתמש

TECHNICAL_SPEC 7.2 מנה בעבר "Head Pose חריג ביחס ל-Session" כדבר שיש לסנן — נוסח שערבב סף מוחלט (שכן מסונן) עם סטייה יחסית לשאר הכיול (שלא). הוסבר למשתמש שההחלטה המקורית (Stage B, Decision 5) כבר קבעה שסטייה יחסית אינה Gate — תנוחת הראש היא Feature שמודל המבט ב-M2-03 אמור ללמוד ממנה, לא רעש שיש לסנן; סינון לפיה היה מקטין את מרחב האימון ומגדיל Over-rejection. המשתמש אישר לעדכן את הספק. TECHNICAL_SPEC.md §7.2 עודכן: הבהיר שהסף המסונן הוא Head Pose מוחלט בלבד, והוסיפה הערה מפורשת שהסטייה היחסית נחשפת כ-Metric תיאורי (head_pose_spread()) ואינה Gate.

Definition of Done

בסוף הכיול יש Dataset מלא של Features ↔ Screen targets שניתן לאמן עליו מודל. [הושג ב-Stage A]

Legacy mapping

M2-T03 + חלק מהפער שזוהה לאחר M2-T02

M2-03 — Gaze Mapping Engine

Status: 🟡 PARTIAL / IN PROGRESS — M2-03A Base Mapping ו־M2-03B Local Correction Foundation מומשו ואומתו אוטומטית; כיול מצלמה אמיתי schema 2 יצר Candidate וה־Live check חשף הסטה מהותית. המועמד נדחה ואין מודל פעיל; נוספה Live Validation לפני Promotion, בחירת מודל שמרנית לפי X/Y והשוואת Linear/Polynomial חיה. נשארים אימות חי, Accuracy/Filter מכוילים ו־Correction Before/After מול המשתמש

מטרה

ללמד את GAZELINK לתרגם Features של העיניים/הראש לנקודה במסך.

לבצע

Feature vector schema.

Train/validation split.

Baseline mapping model.

Advanced comparison model אחד לפחות.

Model benchmark.

Model serialization.

CalibrationEngine.

GazeEstimator.

Screen geometry compatibility.

Raw normalized X/Y.

Pixel X/Y.

Confidence / reason codes.

תוכנית מימוש מאושרת — 2026-09-02

התוכנית אינה משנה את סדר ה־Roadmap. M2-03A ו־M2-03B הם שלבי ביצוע פנימיים של M2-03: קודם מוכיחים Mapping בסיסי ומדיד, ורק אחריו מוסיפים שכבת תיקון מקומית. הפעלת התיקון באמצעות Blink תלויה ב־M3-03, ו־UX מוצרי נגיש ומלוטש שייך ל־M4.

M2-03A — Base Gaze Mapping

Feature extraction

`src/gazelink/gaze_features.py` יהיה מודול טהור לבניית Feature vector ממוספר באמצעות `FEATURE_SCHEMA_VERSION`.

הסדר המאושר: left/right iris X/Y, left/right openness, head yaw/pitch/roll.

יש לתקן במפורש את כיוון X ההפוך בין שתי העיניים ולתעד את תלות iris Y ב־openness.

תיקון מאושר מאוחר יותר: תלות Y ב־openness נמצאה בעייתית ב־Run החי והוסרה ב־Feature schema 2; Y מבוסס כעת על ציר פינות העין הקבוע. הרשומה המלאה מופיעה בהמשך תחת “Feature schema 2 + contiguous stability windows”.

`confidence=0.5` לא ישמש Feature משום שהוא כרגע קבוע ולא נושא מידע.

Sample שחסר בו Feature נדרש לא ישמש לאימון. Model או Correction מגרסת Feature אחרת יידחו ולא ייטענו בשקט.

Model comparison

`src/gazelink/gaze_model.py` יהיה מודול NumPy טהור ללא I/O.

Baseline: Linear Regression נפרד ל־X ול־Y באמצעות `numpy.linalg.lstsq`.

Advanced: Polynomial Regression מדרגה שנייה עם Ridge על iris X/Y ו־openness; Head Pose לא יורחב פולינומית כדי לא לנפח את מספר ה־Features מול Dataset של 45 דגימות.

Ridge lambda וכל קבוע מספרי חדש יהיו Placeholders מרכזיים ומתועדים כלא מכוילים. אין להוסיף SciPy, scikit-learn, Neural Network או תלות ML גדולה.

המודל המתקדם אינו נבחר רק משום שהוא מורכב יותר. אם תוצאות ה־Benchmark אינן עדיפות בבירור, ברירת המחדל היא ה־Baseline.

Validation / Benchmark

ההשוואה תשתמש ב־Leave-One-Target-Out: בכל סיבוב מאמנים על שמונה Targets ובודקים על ה־Target התשיעי. לאחר ה־Benchmark, המודל הנשמר מאומן מחדש על כל הדגימות.

לכל מודל יש לדווח Median/P95 error במרחב מנורמל ובפיקסלים, Error לכל Target/region ו־Prediction latency.

אין להמציא Pass/Fail מספרי כל עוד D-06 פתוח. המספרים נמדדים ומדווחים.

Engine / contracts

`CalibrationEngine` מבצע `CalibrationSessionResult → Feature extraction → Benchmark → Model selection → CalibrationModel`.

`CalibrationModel` יהיה frozen dataclass serializable עם coefficients ל־X/Y, model type, Feature schema version, ScreenGeometry, זהות מודל יציבה, זמן אימון ו־`to_dict/from_dict`.

`GazeEstimator` יקבל רק את `accepted_observation` שעבר את מדיניות האיכות. אסור להשתמש ב־display observation שעלול להישאר `TRACKED` עם Geometry מרוקנת בזמן Debounce.

Observation ישן או Feature חסר לא יפיקו Coordinate תקין למראית עין. אין להשתמש ב־NaN, אפסים או last-known כ־Sentinel מוסתר.

תחזית מחוץ ל־`[0,1]` נשמרת ב־Raw לצורכי אבחון; Pixel output נחתך לגבולות המסך ומקבל `OUT_OF_RANGE` ו־`CLAMPED_TO_SCREEN`.

Confidence הוא Pass-through בלבד עד שיהיה ביטחון אמיתי של המודל. `valid_for_control=False` תמיד ב־M2-03.

Output semantics

חוזה ה־Gaze יבדיל במפורש בין `raw_normalized` (מודל בסיס), `corrected_normalized` (אחרי Correction ולפני Filter) ו־`filtered_normalized` (אחרי M2-04). אין להשתמש ב־`filtered_normalized` כשם חלופי סמוי ל־Correction.

ב־M2-03A: corrected=raw ו־filtered=corrected. ב־M2-03B: corrected=correction(raw) ו־filtered=corrected. שינוי ב־`domain.py` ובבדיקות החוזה יהיה בבעלות אינטגרציה אחת.

Screen mapping / persistence

יש להוסיף המרה טהורה `normalized_to_pixel(GazePoint, ScreenGeometry) → PixelPoint`, עם Clamp לפני יצירת `PixelPoint`.

מודל תקף רק ל־ScreenGeometry שעבורו אומן. יש לבדוק לפחות Screen ID, dimensions ו־orientation. אין להמציא חישוב DPI לפני אימות החוזה מול Windows והמסך האמיתי.

ה־Dataset והמודל יישמרו תחת `.gazelink/calibration/` כ־`dataset_<UTC>.json`, `model_<UTC>.json` ו־`latest_model.json`.

טעינה תאמת Schema, Model identity ו־Screen geometry. קובץ חסר, פגום או לא תואם פירושו "no model", לא Crash. אין לשמור Raw Frames ואין להכניס Dataset אמיתי ל־Git fixtures.

Runtime / live check

`RuntimeTick` יחשוף את `accepted_observation` בלי להריץ שוב את מדיניות האיכות בשכבת Qt.

מסלול אבחוני כגון `--gaze-check` יבצע Calibration → Training → Persistence → Live raw normalized X/Y + Pixel X/Y, ויסומן במפורש `RAW / UNFILTERED / NOT FOR CONTROL`.

M2-03B — Local Correction Foundation

גבול השלב

שלב זה בונה Correction math טהור, Capture של דגימות יציבות, Persistence, Undo/Reset, כלי אבחון מבוסס Targets ידועים ומדידת Before/After.

הוא אינו טוען ש־Blink או שליטה עצמאית מלאה כבר קיימים. בזמן M2 מפעילים את כלי האבחון באמצעות מפתח/מטפל או מקלדת, כאשר Real OS input נשאר מושבת.

Architecture

`src/gazelink/gaze_correction.py` יהיה מודול ייעודי וטהור ככל האפשר: `Base GazeEstimator → Raw prediction → LocalCorrectionEngine → Corrected unfiltered prediction`.

ה־Base CalibrationModel נשאר ללא שינוי ואינו מאומן מחדש בעקבות Correction.

Correction capture

לא משתמשים ב־Auto-scan להזזת נקודה לפי זיכרון המשתמש כ־Ground Truth.

כלי התיקון מציג Target במיקום ידוע, ממתין להתייצבות, אוסף חלון של כ־300–500ms של `accepted_observation` בלבד, ומאגד את ה־Features באמצעות Median.

Capture ישן, חסר, רועש או לא יציב נדחה עם Reason Code. זמני החלון, מספר הדגימות וספי היציבות הם Placeholders מרכזיים ולא מכוילים.

Correction contract

כל Correction ישמור aggregated feature vector, base raw prediction, true target position, residual X/Y, ScreenGeometry, Base model identity, Feature schema version, timestamp ומטא־דאטה מינימלי של חלון ה־Capture.

`residual = true_target_position - base_raw_prediction`. אין לשמור Raw Frames.

Local conservative correction

ה־Correction מעוגן במרחב תחזיות ה־Raw. לכל Correction רדיוס השפעה מוגבל; מחוץ לרדיוס ההשפעה היא אפס מוחלט.

ההשפעה יורדת עם המרחק, כוללת Shrinkage כדי ש־Correction יחיד לא יחול בעוצמה מלאה, ומוגבלת ב־Maximum offset.

Corrections עקביים באותו אזור יכולים להתחזק בהדרגה. Corrections סותרים מפחיתים השפעה או דורשים Retry ואינם יוצרים קפיצה.

רדיוס, Shrinkage, Maximum offset וסף Conflict יהיו Settings מרכזיים ומתועדים כ־Placeholders.

Candidate verification

Correction חדש נשמר תחילה כ־Candidate ואינו הופך פעיל מיד.

האימות ישתמש ב־Held-out observations חדשים, ועדיף ב־Target נוסף סמוך שלא יצר את ה־Correction. Raw ו־Corrected יחושבו על אותו חלון Validation.

יש למדוד Before/After באזור התיקון ולוודא שאין שינוי מעבר לגבול המותר באזורים רחוקים. אם אין שיפור ברור, מבצעים Reject/Rollback או Retry. ספי הקבלה נשארים Placeholders עד למדידה אמיתית.

Persistence / recovery

Corrections יישמרו תחת `.gazelink/calibration/` כשהם משויכים במפורש ל־Base model identity, Feature schema ו־ScreenGeometry.

Correction לא תואם או קובץ פגום לא ייטענו ולא יקריסו את האפליקציה. כאשר מודל הבסיס נפסל או נבנה מחדש, זהות המודל משתנה וה־Corrections הישנים אינם תקפים עוד.

יש לספק `undo_last_correction()`, `clear_all_corrections()`, `disable_corrections()`, `enable_corrections()` ו־`return_to_base_model()`. Undo/Reset חייבים להישמר נכון גם לאחר Restart.

Diagnostic correction flow

Load/train base model → show Raw gaze → show known correction target → capture stable window → stage Candidate → show nearby validation target → report Before/After → Accept/Reject → show Raw and Corrected simultaneously → verify Undo/Reset.

אין להזיז Cursor אמיתי ואין להפיק Mouse actions ב־M2.

בדיקות נדרשות — M2-03A

Handedness בין העיניים, Feature schema version, Missing-feature rejection, שחזור Dataset ליניארי סינתטי ידוע, Polynomial/Ridge finite ויציב, Leave-One-Target-Out metrics, Ridge shrinkage, Screen conversion/clamping, Stale/missing/mismatched input, Model persistence/קובץ פגום, ושימוש Runtime ב־accepted observation בלבד.

בדיקות נדרשות — M2-03B

Capture משתמש רק ב־vetted observations; חלון לא יציב נדחה; Median aggregation דטרמיניסטי; Pixel Target מומר נכון ל־normalized ground truth; Correction מפחית שגיאה ליד ה־Anchor; ההשפעה אפס מחוץ לרדיוס; Maximum offset נשמר; Corrections סמוכים מצטברים ביציבות; Conflicts אינם יוצרים קפיצה; Candidate אינו פעיל לפני Accept; Validation הוא Held-out; Persistence round-trip; Model/Schema/Screen mismatch; קובץ פגום; Undo/Clear/Disable מחזירים במדויק את Base behavior; Raw prediction אינה משתנה.

סדר מימוש מאושר

1. `gaze_features.py` ובדיקות.
2. `gaze_model.py` ובדיקות.
3. `gaze_engine.py`, חוזי מודל, Screen conversion ו־Persistence.
4. חשיפת `accepted_observation` דרך `RuntimeTick`.
5. Base live check.
6. Benchmark בסיסי על Dataset אמיתי.
7. רק לאחר שיש Baseline מדיד: `gaze_correction.py` ובדיקות מתמטיות.
8. Stable target capture.
9. Correction persistence ו־Undo/Reset.
10. Candidate verification ו־Before/After benchmark.
11. Diagnostic correction flow.
12. pytest, ruff check, ruff format --check, mypy ו־build.
13. Real-camera QA ותרחישי Safety/Recovery רלוונטיים.
14. עדכון M2-03 כאן עם ראיות, מדידות, קבצים, סטיות וסיכונים.

רק שלבים 1 ו־2 מתאימים לעבודה מקבילית לפני שנקבעו החוזים. משלב 3 ואילך העבודה תלויה בממשקים ובאינטגרציה בפועל.

סיכונים מאושרים למעקב

Dataset של 45 דגימות בלבד ו־Overfitting; Correlation בין Head Pose לסדר Targets; תלות iris Y ב־openness; Confidence קבוע; DPI לא מאומת; Capture בזמן תנועת ראש/עיניים; Target או תזמון שגויים; Over-correction מתיקון יחיד; פגיעה באזורים רחוקים; Corrections סותרים; Corrections ישנים לאחר שינוי Calibration/Screen/Camera/Schema; ו־Auto-scan מעייף או Blink טבעי שמפעיל פקודה — שני האחרונים יטופלו רק בשלבי M3/M4 המתאימים.

גבולות Definition of Done

M2-03A משלים את ה־Definition of Done המקורי כאשר Observation חי ומאושר הופך לאחר Calibration ל־Raw normalized X/Y ול־Pixel X/Y מדידים, נשמר ונטען מחדש, ללא Control.

M2-03B נחשב מאומת כאשר כלי האבחון יכול להציג Target ידוע, לאסוף Capture יציב, ליצור Correction מקומי, לאמת אותו על דגימות חדשות, לדווח Before/After, לשמור/לטעון/לבטל/לאפס, ולהוכיח שאין השפעה מחוץ לרדיוס שהוגדר.

M2-03 אינו טוען לנקודה יציבה/מסוננת (M2-04), Windows control (M3), הפעלת Correction באמצעות Blink (M3-03), או UX מוצרי עצמאי ומלוטש (M4).

ראיית תכנון

התוכנית אושרה במפורש על־ידי המשתמש ב־2026-09-02.

בוצע — M2-03A Base Gaze Mapping

נוסף Dataset contract מלא: `CalibrationSessionResult` שומר כעת גם את תשע קואורדינטות ה־Targets בפועל. טעינת Dataset ישן ללא השדה משתמשת ב־Default 9-point targets לתאימות, אך כל Dataset חדש שומר את ה־Labels האמיתיים ולא משחזר אותם בניחוש.

נוסף `src/gazelink/gaze_features.py`: Feature schema ממוספר, תיקון handedness של Right-eye X, נתיבי Training/Live זהים, Missing-feature exclusion וללא שימוש ב־Confidence הקבוע.

נוסף `src/gazelink/gaze_model.py`: Linear baseline, Polynomial degree-2 + Ridge, Leave-One-Target-Out CV, Median/P95 normalized+pixel error, per-target error, latency ומדיניות בחירה שמרנית שחוזרת ל־Linear כאשר Advanced אינו עדיף גם ב־Median וגם ב־P95.

נוסף `src/gazelink/gaze_engine.py`: `CalibrationModel`, זהות SHA-256 יציבה לפי סשן הכיול המדויק ותוכן המודל/Schema/Geometry/Camera, `CalibrationEngine`, `GazeEstimator`, typed rejection ללא Coordinate מזויף, normalized→pixel clamp, JSON persistence אטומי וטעינה Recoverable של קובץ חסר/פגום/לא תואם. לכן סשן כיול חדש מבטל Corrections ישנים גם אם מקדמי הרגרסיה יצאו במקרה זהים.

`GazeSample` מפריד כעת במפורש בין `raw_normalized`, `corrected_normalized` ו־`filtered_normalized`, עם backward-compatible load שבו ערך ישן ללא Correction מקבל identity correction.

`RuntimeTick` חושף `accepted_observation` בנפרד מ־display observation. Base estimator ו־Correction capture צורכים רק אותו.

Guided calibration מלא עוצר את Timer הדגימה מיד עם ה־sample האחרון, שומר Dataset JSON lossless, מריץ Benchmark, מאמן ושומר Model + `latest_model.json` בלי לדרוש סגירה ידנית של החלון. כשל Training/Storage אינו מוחק את הלוג האנושי; סגירת החלון עדיין משחררת את המצלמה דרך מסלול ה־shutdown הקיים.

נוסף `--gaze-check`: Diagnostic full-screen של Raw/Corrected coordinates, מסומן `UNFILTERED / NOT FOR CONTROL`; ללא Cursor או OS input. פאנל ה־Status/Correction הוא Qt Widget נגרר אמיתי, נפתח אוטומטית במרכז, חוזר למרכז גם באמצעות `C`, ונשמר נגיש בתוך גבולות החלון בלי לקחת Focus ממקשי האבחון. התנהגות Press/Move/Release אומתה באמצעות אירועי Qt אמיתיים ב־offscreen test.

בוצע — M2-03B Local Correction Foundation

נוסף `src/gazelink/gaze_correction.py`: Target-based stable-window capture, Median feature aggregation, Sample/Model/Schema/Geometry identity, compact-support local residual correction, Hard zero מחוץ לרדיוס, Shrinkage, Maximum offset, Conservative conflict rejection, Candidate staging, Held-out frame validation, Persistence, Enable/Disable, Undo ו־Clear.

נוסף `src/gazelink/correction_diagnostic.py`: State machine טהור של Stabilize → Capture → nearby held-out Validation target → Review → Accept/Reject. Capture ו־Validation אינם יכולים לשתף Frame IDs.

`--gaze-check` משלב Operator diagnostic בלבד: מקשים 1–9 בוחרים Target ידוע; `A` מאשר רק Candidate שהוריד Error; `X` דוחה; `U` מבטל אחרון; `R` מאפס; `D` מכבה/מחזיר Corrections. Blink/Voice/OS input אינם מחוברים בשלב זה.

קבצים שהשתנו — Implementation

`src/gazelink/domain.py`, `src/gazelink/calibration.py`, `src/gazelink/runtime.py`, `src/gazelink/calibration_window.py`, `src/gazelink/app.py`.

קבצים חדשים — Implementation

`src/gazelink/gaze_features.py`, `src/gazelink/gaze_model.py`, `src/gazelink/gaze_engine.py`, `src/gazelink/gaze_correction.py`, `src/gazelink/correction_diagnostic.py`, `src/gazelink/gaze_window.py`.

בדיקות חדשות/מעודכנות

נוספו `tests/test_gaze_features.py`, `tests/test_gaze_model.py`, `tests/test_gaze_engine.py`, `tests/test_gaze_correction.py`, `tests/test_correction_diagnostic.py`, `tests/test_gaze_window.py`.

עודכנו `tests/test_domain.py`, `tests/test_calibration.py`, `tests/test_calibration_window.py`, `tests/test_runtime.py`, `tests/test_smoke.py`.

אימות אוטומטי — 2026-09-02

Baseline לפני השינוי: 236 passed, 2 deselected.

לאחר המימוש ועדכון פאנל האבחון: 276 passed, 2 deselected; Coverage כולל 86%.

`ruff check .` עבר. `ruff format --check .` עבר לאחר Formatting. `mypy` strict עבר על 57 Source/Test files. `python -m build` בנה בהצלחה sdist ו־wheel. ריצת `scripts/run_quality.ps1` עברה Pytest/Ruff/Format/Mypy אך ניסיון Build מבודד ראשון נכשל משום שה־Sandbox חסם הורדת `setuptools>=75`; אותה פקודת Build הורצה מחדש עם גישת Dependency מאושרת והצליחה.

Scenario QA אוטומטי

Happy-path training/estimate; Rejected/Missing/Stale/Future observation ללא Coordinate; Geometry mismatch; Out-of-range raw + clamped pixels; Corrupt model/dataset recovery; Runtime debounce שאינו חושף Geometry ישנה; Stable/short/unstable Correction capture; Pixel ground-truth conversion; Local improvement; Exact zero מחוץ לרדיוס ובגבול; Multiple consistent corrections; Max offset; Conflicts; Candidate inert before validation; Held-out Frame enforcement; Persistence/model mismatch; Undo/Clear/Disable; Diagnostic Accept/Reject/Undo/Clear state transitions; Camera loss/recovery/shutdown מה־Suite הקיים; Real OS input אינו קיים ב־`src` ונשאר מושבת.

מדידות דטרמיניסטיות — Dataset סינתטי ידוע, לא נתוני מוצר

Linear Dataset ידוע: נבחר `LINEAR`; Baseline Median/P95 = 0.000 normalized ו־0.000px; Median prediction latency ≈0.0035ms. Advanced Median = 0.000610 normalized / 0.972px, P95 = 0.001728 / 2.690px, latency ≈0.0065ms.

Local Correction property benchmark: Error normalized ליד ה־Anchor ירד מ־0.200000 ל־0.100000; Distant shift מחוץ לרדיוס = 0.000000.

המספרים לעיל מוכיחים Correct wiring/properties בלבד. הם אינם Accuracy של המשתמש ואינם סוגרים את D-06.

Real-camera QA — 2026-09-02 — נכשל ונדרש תיקון לפני אימות M2-03

שני Sessions חדשים על מסך `LS49C95xU` ‏4096×1152 הגיעו טכנית ל־`COMPLETE`, אך אינם מהווים כיול תקף. ב־Session האחרון נשמרו 45 accepted מול 111 rejected; מצבי ה־Runtime היו 101 `LOST`,‏ 10 `LOW_CONFIDENCE` ו־45 `TRACKED`. כל חמש הדגימות של Target נאספו בתוך כ־126–160ms, והדגימה הראשונה של ה־Target הבא התקבלה רק כ־31–34ms אחרי האחרונה של הקודם. לכן אין זמן אנושי להתייצב ולהעביר מבט, וניתן להשלים את כל הכיול תוך הסתכלות לכיוון המצלמה במקום אל ה־Targets.

Benchmark אמיתי של `dataset_20260902T071505500033Z.json` אישר שהמודל שנוצר אינו שימושי: Linear Leave-One-Target-Out Median ‏1512.90px ו־P95 ‏2835.51px; Advanced Median ‏1543.49px ו־P95 ‏2843.07px. `latest_model.json` הנוכחי הוא Artifact אבחוני מה־Session הכושל ואינו מודל מוצר מאומת.

הכשל המוכח הוא היעדר Stabilization/Dwell לפני Capture לכל Target. חשדות נוספים שמחייבים Observability ואימות לפני שינוי ספים: ה־Runtime המשותף מפעיל Head Pose limits של Live ‏(30° yaw) לפני ה־Calibration gate הנדיב ‏(55°), Reason codes פנימיים נמחצים בלוג ל־`tracking_not_ready`, ו־Targets אופקיים ב־5%/95% קיצוניים במיוחד למסך 49-inch Ultrawide. אין לנסות לפצות על Base calibration זה באמצעות Local Correction.

Live gaze / correction QA follow-up — 2026-09-02

המשתמש דיווח שגם במבט ממוקד לנקודה קבועה ה־Raw gaze רוקד ללא הפסקה. זה תואם לחוזה הנוכחי: ב־M2-03 `filtered_normalized=corrected_normalized`, ולכן כל רעש Frame-to-Frame מוצג ללא Filter; Local Correction מזיז Bias מקומי ואינו מפחית Jitter. במודל האחרון יש גם מקדמים גדולים ו־Cross-validation גרוע, ולכן הוא עלול להגביר רעש קטן ב־iris/head features לתנועה גדולה בפיקסלים על מסך 4096px.

זרימת המקשים `1–9` נמצאה לא־discoverable: המיפוי הנוכחי הוא שורות משמאל לימין (`1/2/3` למעלה, `4/5/6` באמצע, `7/8/9` למטה; `5` הוא מרכז), אך הוא אינו מוצג למשתמש לפני הבחירה. `latest_corrections.json` לאחר הבדיקה מכיל אפס Corrections ו־`enabled=false`, ולכן הריקוד שנצפה אינו נגרם מתיקון פעיל.

התוכנית אושרה ומומשה חלקית ללא שינוי סדר ה־Roadmap: קודם תוקנה אמינות איסוף ה־Calibration, אחר כך חוברה שכבת ה־Filtering האבחונית של M2-04, ורק לאחר Calibration חי תקף חוזרים לאימות Correction.

תיקון Calibration + Stability foundation — 2026-09-02

`CalibrationUiController` כולל כעת State מפורש של `STABILIZING`/`COLLECTING`: ברירת המחדל ממתינה 750ms לאחר הופעת כל Target, אוספת לאורך לפחות 500ms ומרווחת accepted samples ב־80ms לפחות. לכן חמשת הפריימים אינם יכולים עוד להשלים Target בתוך 126–160ms, והמעבר ל־Target הבא פותח מחדש חלון התייצבות. כל הערכים הם Placeholders שדורשים כיוון מול מצלמה אמיתית.

Runtime הכיול משתמש כעת ב־Head Pose limits הייעודיים של Calibration ‏(55°/40°/40°) וב־sample age של 500ms, במקום להיחתך קודם על ידי ברירות המחדל המחמירות של Live. למסך ביחס רוחב 2.40 ומעלה Grid הכיול משתמש ב־15%/50%/85% אופקי וב־5%/50%/95% אנכי; גם סף ה־Ultrawide וה־inset הם Placeholders.

נוסף Quality promotion gate למודל: Linear/Advanced שנבחר חייב לנצח Predictor טריוויאלי שמחזיר תמיד את מרכז המסך גם ב־Median וגם ב־P95 של held-out targets. Dataset נשמר לצורכי אבחון, אבל Candidate שנכשל אינו נכתב כ־`latest_model.json`; מודל פעיל קודם נשאר ללא שינוי כדי שניסיון חדש וכושל לא ימחק כיול תקף. בדיקה חוזרת של `dataset_20260902T071505500033Z.json` נכשלה בשער עקב P95 ‏2835.51px מול Center baseline ‏1914.16px. ה־latest הישן, שכבר הוכח ככושל, הועבר ידנית ובאופן recoverable ל־`rejected_latest_model_20260902T083654538026Z.json`, ולכן `--gaze-check` לא יטען אותו בטעות.

נוסף `src/gazelink/gaze_filter.py`: מימושים טהורים/resettable של EMA, One Euro ו־Kalman, מדדי Stationary jitter ב־pixels ו־Rolling window של שתי שניות. `--gaze-check` מפעיל כרגע One Euro כברירת מחדל אבחונית, מציג Raw cyan מול Filtered green ואת `jitter_p95(raw, filtered)`, ומאפס Filter+metrics מיד כשאין accepted observation. הבחירה והספים טרם כוילו מול המשתמש ולכן אינם החלטת מוצר סופית.

UX התיקון אינו מקבל עוד מספר עיוור: `K` מציג מפת 3×3 ממוספרת במיקומים האמיתיים; רק כשהמפה פתוחה `1–9` בוחר Target. בזמן Stabilize/Capture/Validation נקודות Raw/Filtered מוסתרות כדי שהמשתמש לא ירדוף אחרי רעש המודל; Target הידוע נשאר גלוי. Before/After, Accept/Reject, Undo/Reset ו־Disable נשארו כפי שהוגדרו.

QA אוטומטי: 282 passed, 2 deselected; coverage 86%; `ruff check .`, `ruff format --check .`, `mypy src` ו־`python -m build` עברו. בדיקות חדשות מכסות timing/target transition, Ultrawide grid, דחיית מודל לא־אינפורמטיבי תוך שמירת מודל פעיל קודם, quarantine recoverable למודל הכושל, הפחתת frame-jump בכל שלושת המסננים על אותו רצף סינתטי, ו־reset ללא replay לאחר Tracking loss. לא בוצעה טענה ל־Accuracy או Jitter אמיתיים מול מצלמה אחרי השינוי.

Gaze feature preflight diagnostic — 2026-09-02

לאחר ששתי ריצות Calibration מתוזמנות חדשות נכשלו גם ב־Training error וגם ב־held-out error, נוסף מצב אבחון ייעודי לפני אימון: `python -m gazelink --gaze-feature-check`. הוא מציג חמישה Targets בסדר CENTER → LEFT → RIGHT → UP → DOWN, ממתין 1,000ms להתייצבות ואוסף 1,500ms של `accepted_observation` בכל כיוון, עם לפחות 10 דגימות מרווחות. Tracking loss מוחק את החלון החלקי ומתחיל Stabilization מחדש כדי לא לערבב שתי תנוחות פנים.

`src/gazelink/feature_check.py` הוא State machine ומנתח טהור ללא Qt/Camera/Model: הוא מחשב Median לכל תשעת ה־Features, ‏P95 stationary spread, הפרש כל עין מול Center ו־Verdict מפורש לכל כיוון (`PASS`, `NO_SEPARATION`, `INVERTED`, `EYES_DISAGREE`). הספים הם Placeholders אבחוניים בלבד ואינם Product accuracy gate.

`src/gazelink/feature_check_window.py` הוא חלון Full-screen נפרד שאינו טוען או מאמן CalibrationModel ואינו מפעיל OS input. בסיום הוא כותב JSON lossless וטקסט אנושי תחת `.gazelink/feature_checks/`; נשמרים רק scalar feature vectors, Frame IDs וזמנים — ללא Frame/image/landmark array. `R` מתחיל מחדש ו־`Esc` מבטל וסוגר.

QA לאחר ההוספה: 287 passed, 2 deselected; coverage 85%; `ruff format .`, `ruff check .`, `mypy src`, `python -m gazelink --help` ו־`python -m build` עברו. בדיקות מכסות Stabilization/Collection, מחיקת חלון לאחר Tracking loss, תנועה תקינה בכל ארבעת הכיוונים, זיהוי כיוון הפוך, מניעת ביטול הדדי של רעש הפוך בין שתי העיניים, Persistence של דוח scalar-only וחשיפת Flag ה־CLI. חיפוש ב־`src` לא מצא SendInput/SetCursorPos/mouse_event/pyautogui/win32api/ctypes.windll. נדרש כעת Run אמיתי אחד מול המשתמש וניתוח הדוח לפני שינוי נוסף ב־Feature extraction או ב־Calibration model.

Feature schema 2 + contiguous stability windows — 2026-09-02

ה־Run החי `feature_check_20260902T090507306743Z.txt` נכשל באופן אינפורמטיבי: LEFT עבר עם `combined_delta=-0.20018`, אך RIGHT לא עבר משום שבעין ימין ההפרדה הייתה רק `+0.02398` מול סף `0.03`; UP/DOWN כמעט לא יצרו הפרדה אנכית (`+0.01326`/`+0.00839`). CENTER ו־LEFT גם היו רועשים (`stationary_p95=0.27743`/`0.14485`), בעוד RIGHT/UP/DOWN היו מתחת לסף `0.08`. הדוח הישן לא מדד בנפרד תנועת Head Pose בתוך החלון.

תוקן חוזה ה־Vertical iris feature: ‏Y אינו עוד יחס בין העפעף העליון לתחתון, משום שעפעפיים שנעים עם הקשתית יכולים לבטל את האות האנכי. ב־Feature schema 2 הוא היסט ניצב מציר פינות העין, מנורמל ברוחב העין ומכוון כך שחיובי הוא כלפי מטה. `FEATURE_SCHEMA_VERSION` הועלה ל־2, ולכן Dataset/Model/Correction מסכמה 1 נדחים במפורש ולא נטענים לפי משמעות שגויה.

נוסף Primitive משותף שמודד P95 של Worst-eye iris spread ושל תנועת yaw/pitch/roll, ובוחר את תת־החלון הרציף והיציב הטוב ביותר. `--gaze-feature-check` שומר רק חלון כזה; אם אין חלון מתאים הוא מוחק את הניסיון החלקי, חוזר ל־Stabilization על אותו כיוון ומבקש Retry. הדוח החדש מציג גם `head_pose_p95_deg` לכל כיוון.

איסוף ה־Guided calibration אינו מאשר עוד את הדגימות הראשונות מיד ואינו בונה מהן Baseline שעלול להיות מורעל. דגימות שעברו Gates נשארות provisional עד סוף חלון ה־Capture; רק תת־חלון רציף בגודל `min_samples_per_target` שעומד ב־`max_stable_eye_p95=0.08` וב־`max_stable_head_pose_p95_deg=5.0` נשמר כ־accepted. מועמדים מחוץ לחלון נשמרים כ־`target_outlier`; ללא חלון יציב אותו Target מתחיל מחדש אוטומטית. Tracking/visibility/confidence/age/head-pose failure באמצע חלון מפסיק אותו ומונע ערבוב משני מצבי מעקב. שני הספים הם Placeholders שטרם כוילו מול המשתמש.

QA אוטומטי לאחר התיקון: 296 passed, 2 deselected; coverage 85%. `ruff format .`, `ruff check .`, `mypy src tests`, `python -m gazelink --help` ו־`python -m build` עברו. ה־Build המבודד הראשון נכשל רק משום שה־Sandbox חסם הורדת `setuptools>=75`; אותה פקודה הורצה מחדש עם גישה מאושרת והצליחה לבנות sdist ו־wheel. בדיקות חדשות מכסות אות Y שנשמר גם כשהעפעפיים נעים עם הקשתית, פסילת Schema שגוי, בחירת suffix רציף יציב בלי תפירת פריימים, דחיית רעש עיניים/תנועת ראש, Retry אוטומטי באבחון ובכיול, ומניעת Bootstrap מורעל. לא בוצע עדיין Run מצלמה לאחר שינוי Schema 2 ולכן M2-03 נשאר PARTIAL.

סדר האימות המעודכן: קודם להריץ שוב `python -m gazelink --gaze-feature-check`. רק אם CENTER/LEFT/RIGHT/UP/DOWN יציבים וכל ארבעת כיווני ההפרדה עוברים, להריץ `python -m gazelink --guided-calibration`. רק Calibration מסכמה 2 שעובר את Model promotion gate יכול לפרסם `latest_model.json`; אחריו ממשיכים ל־`--gaze-check` ול־Correction Before/After.

Real-camera Feature schema 2 rerun — `feature_check_20260902T092959481528Z` — 2026-09-02

השינוי פתר את כשל היציבות ואת כיוון האות האנכי, אך ה־Preflight עדיין מדווח `Overall diagnostic pass: False` בגלל סף הפרדה אנכי גבולי. כל חמשת הכיוונים יציבים: Eye `stationary_p95` נע בין `0.02357` ל־`0.03365` מול סף `0.08`, ו־`head_pose_p95_deg` נע בין `1.10002°` ל־`1.85321°` מול סף `5°`. בהשוואה ל־Run הישן, CENTER השתפר מ־`0.27743` ל־`0.03244` ו־LEFT מ־`0.14485` ל־`0.03365`.

LEFT ו־RIGHT עוברים כעת (`-0.08363` ו־`+0.05677`). גם UP/DOWN נעים כעת בכיוון הנכון ובעקביות בין שתי העיניים: UP הוא `-0.03363`/`-0.02611`, ו־DOWN הוא `+0.02416`/`+0.02919`. הם נכשלים רק משום שמדיניות ה־Verdict דורשת כרגע `abs(delta) >= 0.03` בכל עין; ה־combined deltas הם `-0.02955` ו־`+0.02698`. סטיית התקן האנכית בתוך החלונות היא בערך `0.009–0.011` ברוב העיניים, כך שההפרדה שנמדדה היא בערך פי `2.3–3.5` מהרעש התוך־חלוני ולא נראית כהיעדר אות.

מסקנה: אין לחזור להגדרת Y הישנה ואין עדיין להריץ Guided calibration כאשר ה־Gate הרשמי נכשל. השינוי הבא המומלץ הוא סף נפרד לציר האנכי (`0.02` כ־Placeholder מדוד ראשון) תוך שמירת הסף האופקי `0.03`, ולא הורדה גלובלית של הסף. לאחר שינוי כזה נדרש Feature-check חי נוסף אחד לפני Calibration.

Axis-specific separation thresholds — בוצע ואומת אוטומטית — 2026-09-02

`FeatureCheckSettings.min_direction_delta` הוחלף תחילה בספים נפרדים לצירים. לאחר ה־Run המבוקר, המדיניות הסופית הוחמרה/דוקדקה כמתועד תחת “Binocular vertical verdict” להלן; הרשומה כאן נשמרת כהיסטוריית ההחלטה ולא כחוזה הנוכחי.

Replay דטרמיניסטי של כל 50 הדגימות מה־Run החי `feature_check_20260902T092959481528Z.json` דרך המנתח המעודכן מחזיר `overall_pass=True`, ‏`LEFT/RIGHT/UP/DOWN=PASS`, וכל חמשת הכיוונים Stable. זה מוכיח שהנתונים שכבר נאספו עומדים במדיניות החדשה, אך עדיין נדרש Run חי נוסף כדי לאמת Reproducibility לפני Guided calibration.

QA לאחר השינוי: 297 passed, 2 deselected; coverage 85%. `ruff format .`, `ruff check .`, `mypy src tests`, Smoke בטוח (`camera_enabled=false`, ‏`os_input_enabled=false`) ו־`python -m build` עברו; נבנו sdist ו־wheel. לא הופעל OS input ולא הורץ Camera כחלק מה־QA האוטומטי.

Real-camera reproducibility rerun — `feature_check_20260902T095017431662Z` — 2026-09-02

הקבצים תקינים ומכילים 50 Samples עם 50 Frame IDs ייחודיים. כל חמשת חלונות ה־Capture יציבים מאוד: Eye `stationary_p95=0.00980–0.03724` ו־Head `p95=0.88761°–2.61135°`, ולכן הכשל אינו Jitter בתוך Target. למרות זאת `Overall diagnostic pass=False`: ‏RIGHT עבר; LEFT היה גבולי ובעין שמאל הגיע רק ל־`-0.02411` מול הסף האופקי `0.03`; ‏UP היה הפוך בשתי העיניים (`+0.02716/+0.03141`); ‏DOWN לא יצר הפרדה (`+0.00894/+0.01230`).

בין Targets נמדדה תנועת ראש משמעותית ביחס ל־CENTER, אף שהראש היה יציב בתוך כל Target: ‏LEFT ‏yaw `+7.30°` ו־roll `-8.51°`; ‏DOWN ‏pitch `-8.24°` ו־roll `-6.74°`. מאחר ש־`iris_in_eye` הוא אות יחסי לראש, תנועה בין Targets יכולה לגרום לפיצוי של העיניים ולשנות או להפוך את האות שה־Preflight מנסה לבודד. ה־Run הקודם באותה Feature schema נתן UP/DOWN בכיוון הנכון ועובר ב־Replay עם הספים החדשים, ולכן אין ראיה שמצדיקה הורדת ספים נוספת או חזרה לחוזה Y הישן.

החלטת המשך: Guided calibration עדיין חסום. נדרש Run מבוקר נוסף שבו הפנים נשארות מכוונות למרכז והמעבר בין Targets נעשה בעיניים בלבד. אם גם הוא אינו Reproducible, יש לשפר את כלי ה־Preflight כך שיזהה וידווח במפורש `HEAD_MOVED_BETWEEN_TARGETS` או לבחון מדד שמשלב Head Pose; אין להחיל Gate כזה על Calibration עצמו בלי החלטה חדשה, משום ששם Head Pose הוא Feature חוקי של המודל.

Controlled-head real-camera rerun — `feature_check_20260902T095403086523Z` — 2026-09-02

ה־Run המבוקר תקין ומכיל 50 Samples ו־50 Frame IDs ייחודיים. תנועת הראש בין Targets ירדה משמעותית: מרחק תנוחה מ־CENTER הוא `0.90°` ב־LEFT, ‏`1.69°` ב־RIGHT, ‏`2.50°` ב־UP ו־`1.57°` ב־DOWN. כל חמשת החלונות Stable עם Eye `stationary_p95=0.01525–0.06149` ו־Head `p95=0.44987°–1.19938°`.

LEFT, RIGHT ו־DOWN עברו בבירור. UP נע בכיוון הנכון בשתי העיניים אך קיבל `NO_SEPARATION`: ‏left `-0.01302`, ‏right `-0.02057`, ‏combined `-0.01760` מול סף אנכי `0.02` שנדרש כרגע מכל עין בנפרד. ניתוח 10 הדגימות לכל חלון מראה שהפרש הממוצעים CENTER→UP הוא בערך `-0.0106` בעין שמאל עם pooled within-window noise של כ־`0.008`, ובעין ימין בערך `-0.0190` עם noise של כ־`0.008`; כלומר שתי העיניים מסכימות בכיוון, אך עין שמאל מספקת אות חלש יותר.

מסקנת הביניים הייתה שיציבות האיסוף ותיקון Y מאומתים, ואין הצדקה להוריד עוד סף אחיד לכל עין. הומלץ על Verdict דו־עיני עם סף Combined, סימן עקבי ורצפת אות לכל עין. ההחלטה אושרה ומומשה מיד לאחר מכן כמתועד להלן.

Binocular vertical verdict — בוצע ואומת — 2026-09-02

ה־Gate האופקי נשאר שמרני ודורש מכל עין `abs(delta) >= 0.03`. ה־Gate האנכי דורש כעת `abs(combined_delta) >= 0.015`, ‏`abs(delta) >= 0.01` בכל עין, ושתי העיניים חייבות להסכים בסימן; מנגנוני `INVERTED` ו־`EYES_DISAGREE` נשארו פעילים. הספים נשמרים ב־JSON ומודפסים בדוח כ־`vertical_combined_delta` ו־`vertical_per_eye_delta`.

Replay של ה־Run המבוקר `feature_check_20260902T095403086523Z.json` מחזיר כעת `overall_pass=True` וכל ארבעת הכיוונים `PASS`. Replay של ה־Run הבעייתי `feature_check_20260902T095017431662Z.json` עדיין מחזיר `overall_pass=False`: ‏LEFT `NO_SEPARATION`, ‏RIGHT `PASS`, ‏UP `INVERTED`, ‏DOWN `NO_SEPARATION`. בכך הוכח שהשינוי מקבל אות דו־עיני חלש־אך־עקבי בלי להלבין Run שבו הכיוון הפוך או ההפרדה זניחה.

QA מלא: 299 passed, 2 deselected; coverage 85%. `ruff format .`, `ruff check .`, `mypy src tests`, Smoke בטוח ו־`python -m build` עברו; נבנו sdist ו־wheel. לא הופעל OS input. ה־Preflight החי האחרון עבר ב־Replay תחת החוזה המאושר ולכן החסימה על Guided calibration הוסרה; השלב הבא הוא Session כיול חדש מסכמה 2.

Live validation gate אחרי Calibration — בוצע אוטומטית, נדרש אימות מצלמה אמיתי — 2026-09-02

ה־Session החי `dataset_20260902T100240835324Z.json` השלים 45 דגימות accepted (5 לכל Target) ו־31 rejected. ה־Linear candidate ניצח את Center baseline ב־Leave-One-Target-Out ולכן עבר את שער ה־offline הקיים: Median `210.8px`, P95 `1253.1px`, מול Center Median `1433.2px`, P95 `1524.0px`, latency `0.0035ms`. עם זאת, Error ה־held-out אינו אחיד: Top-center היה `835.1px` ו־Top-right `1289.7px`; המדדים אינם Accuracy מוצרית מאושרת.

ב־`--gaze-check` המשתמש דיווח שהמבט למעלה לא נתפס, ושגם כאשר הוא מסתכל למרכז הנקודה הולכת ימינה. זו הוכחה שה־offline promotion לבדו אינו מספיק וש־Local Correction לא אמור לכסות הטיה בסיסית זו. ה־`latest_model.json` הפעיל הועבר באופן recoverable אל `rejected_latest_model_20260902T102524505088Z.json`; שום OS input לא הופעל.

נוספו `live_validation.py` ו־`validation_window.py` ו־CLI `--gaze-validation`. מעתה `--guided-calibration` שומר Dataset ו־`candidate_model_<UTC>.json`, אך אינו כותב `latest_model.json`. ה־Candidate נטען רק ב־`--gaze-validation`, שמציג חמש נקודות ידועות עם מדידות חדשות: CENTER, UP_CENTER, UP_RIGHT, LEFT_CENTER ו־RIGHT_CENTER. ארבע נקודות אינן Labels של ה־9-point training grid; CENTER נשמרת כדי לזהות בדיוק את ה־offset החי שעליו דיווח המשתמש. כל נקודה אוספת חמש תחזיות Raw חדשות לאחר Stabilization ומדווחת Median/P95 error בפיקסלים. Tracking withheld מאפס את חלון הנקודה; אין Coordinate מזויף.

אין סף Pass/Fail מספרי חדש, בהתאם ל־D-06 הפתוח: לאחר כל המדידות המשתמש/מפעיל רואה את הערכים ולוחץ `A` כדי לפרסם את ה־Candidate כ־`latest_model`, או `X` כדי לדחות ולארכב את manifest ה־pending. עד לאישור מפורש, מודל פעיל קודם נשמר ללא שינוי; Candidate לא תקף לעולם ל־`--gaze-check` או Control. כך מודל שנראה טוב ב־benchmark אך זז ימינה בלייב אינו הופך למודל פעיל בשקט. בסיום כל Run של Validation נכתבים אוטומטית `.gazelink/validation_logs/validation_<UTC>.txt` ו־`.json` עם Model ID, Geometry, Targets, תחזיות Aggregate ו־Median/P95 בלבד; הנתיבים מודפסים למסוף לפני `A`/`X`. אין בדוח frames, landmarks או Features גולמיים.

QA אוטומטי לאחר השינוי: `306 passed, 2 deselected`, coverage כולל `85%`; `ruff check .`, `ruff format --check .`, `mypy src tests`, `python -m gazelink --help` ו־`python -m build` עברו. נוספו בדיקות ל־pending/promotion שלא דורסים מודל פעיל לפני אישור, למדידת offset חי של `383.8px`, לאיפוס חלון לאחר withheld tracking, ל־CLI החדש ולדוח Validation shareable ללא iris/landmarks. לא בוצע עדיין Run מצלמה של `--gaze-validation`, ולכן M2-03 נשאר PARTIAL.

בחירת מודל שמרנית לפי צירים — בוצע אוטומטית, נדרש אימות מצלמה אמיתי — 2026-09-02

דוח ה־Live validation של ה־Candidate `3c7f8996…` הראה היסט Y עקבי כלפי מעלה: CENTER נחזה ב־`(0.51, 0.18)` במקום `(0.50, 0.50)`, שגיאת Median `367px`; גם LEFT/RIGHT_CENTER נחזו ב־Y `0.21/0.27` במקום `0.50`. X היה קרוב ליעדים. ה־Candidate נשאר pending ואינו פעיל. התיקון אינו Local Correction, כי מדובר בהטיית Base map גלובלית.

`ModelMetrics` מודד כעת גם Median/P95 של absolute X ושל absolute Y במרחב normalized, עבור Center baseline, Linear ו־Polynomial. Polynomial נבחר רק אם הוא משפר את השגיאה המשולבת ואינו מחמיר אף אחד מארבעת מדדי X/Y; אחרת נבחר Linear. אין זה סף Accuracy מוצרי חדש ואינו מבטל Live validation. סיכום הקליברציה למסוף מדפיס כעת Linear/Advanced X/Y Median+P95, כדי שהבחירה תהיה ניתנת לסקירה.

Replay על `dataset_20260902T102939128973Z.json` תחת המדיניות החדשה בחר `LINEAR`: ל־Polynomial הייתה אמנם שגיאה משולבת טובה יותר, אך P95 X שלו היה `0.134` מול `0.109` ב־Linear. הבדיקה אינה מוכיחה שה־Linear מדויק בלייב; היא מוכיחה שהבחירה לא תקריב ציר אחד ללא הצגה. נוספה בדיקה שמוודאת שיפור משולב לא בוחר Advanced כאשר Y מורע, ובדיקה לסיכום X/Y. טרם הורץ כיול מצלמה חדש לאחר שינוי המדיניות.

השוואת מודלים ב־Live Validation — בוצע אוטומטית, נדרש אימות מצלמה אמיתי — 2026-09-02

וולידציית המועמד `a6b529d6…` שיפרה את CENTER ל־Median `85px` ואת LEFT_CENTER ל־`72px`, אך UP_CENTER/UP_RIGHT נשארו עם `262px`/`288px` Median וה־CENTER P95 היה `1417px`. המועמד נדחה במפורש, manifest הועבר ל־`rejected_pending_validation_20260902T105906722345Z.json`, ו־`latest_model.json` אינו קיים.

כיול חדש שומר מעתה את Linear ואת Polynomial מאותו Dataset תחת manifest pending אחד; המועמד הישן בעל קובץ יחיד נשאר נתמך. `--gaze-validation` מריץ את שני המודלים באותם Frames חיים ובאותם חלונות Stabilization/Capture. אם Tracking withheld, שני החלונות מתאפסים יחד. הדוח החדש `validation_comparison_<UTC>.txt/.json` מכיל רק מזהי מודל, labels, Targets ותחזיות/Median/P95 aggregate — ללא Frames, Landmarks או Features. המלצה אוטומטית קיימת רק אם מודל אחד אינו גרוע יותר ב־Median וב־P95 בכל Target ונעשה טוב יותר לפחות פעם אחת; `A` פעיל רק אז ומפרסם רק את אותו Model ID. אין המלצה פירושה דחייה בטוחה, לא בחירה סמויה לפי Benchmark.

QA אוטומטי: `308 passed, 2 deselected`, coverage `84%`; `ruff format --check .`, `ruff check .`, `mypy src tests`, `python -m gazelink --help` ו־`python -m build` עברו. בדיקות חדשות מכסות השוואה סימולטנית, המלצה רק לדומיננטי בכל Target, דוח comparison aggregate-only, שמירת/טעינת שני מועמדים וקידום מפורש של מועמד שאינו בחירת ה־offline. טרם הורץ כיול מצלמה חדש אחרי הרחבה זו.

מועמדי Feature-profile + Feature drift ב־Validation — בוצע אוטומטית, נדרש אימות מצלמה אמיתי — 2026-09-02

נוספו שני מועמדי Linear בעלי חוזה Feature מפורש: `AXIS_IRIS` משתמש ב־iris X וב־yaw לציר X וב־iris Y וב־pitch לציר Y; `AXIS_IRIS_LIDS` מוסיף רק לציר Y את שני מדדי openness. שני הצירים עוברים נרמול שנשמר יחד עם המודל. הבחירה אינה עוברת בשקט ל־Feature נוסף: שני ה־Linear candidates נמדדים ב־Validation חי על אותם Frames, ואילו Polynomial נכנס להשוואה רק אם כבר ניצח שמרנית את ה־offline baseline. לכן כל מועמד שניתן לקידום נמדד בפועל, ואין תלות ב־Benchmark בלבד.

דוח Validation כולל מעתה לכל Target גם תקציר Feature-drift מינימלי: Median של דגימות ה־live מול Anchor אינטרפולטיבי מה־Dataset, עם שם Feature בעל הסטייה המנורמלת הגדולה ביותר. הוא נשאר מדד אבחוני בלבד ואינו Gate לקידום. הדוח אינו שומר Frames, Landmarks, Features גולמיים או נתוני Biometric per-frame. אם Dataset ההפניה חסר, ה־Validation ממשיך ללא מדד זה.

תוקנה התנגשות Persistence: כמה candidate artifacts הנשמרים באותו UTC tick מקבלים suffix רציף במקום לדרוס זה את זה; manifest של השוואה תמיד מפנה לקבצים נפרדים. QA: `315 passed, 2 deselected`; `ruff check .`, `ruff format --check .`, `mypy src tests`, `python -m gazelink --help` ו־`python -m build` עברו. לא הופעל OS input ולא בוצע Run מצלמה אחרי השינוי.

אבחון Y-axis root cause + הסרת קונפאונד Head-pitch — בוצע אוטומטית, נדרש Live validation מצלמה אמיתי — 2026-09-02

לאחר ש־`--gaze-validation` על ה־Candidate מ־`dataset_20260902T114309350044Z.json` הראה שגיאת Y שיטתית (CENTER נחזה `(0.48, 0.64)` במקום `(0.50, 0.50)`; UP_CENTER `(0.49, 0.38)` במקום `(0.50, 0.18)`), בוצע Audit מלא של השרשרת Feature→Train→Validate לפני כל שינוי קוד, לפי דרישת המשתמש שלא לנחש.

נבדק סמנטית כל Feature (`gaze_features.py`): כיוון Y מתועד ואומת כ"חיובי כלפי מטה" גם ב־Raw data — Median של `left_iris_y`/`right_iris_y` ב־45 הדגימות המתקבלות עולה מונוטונית UP→CENTER→DOWN (‏`0.217→0.272→0.327` שמאל), כנ"ל pitch. נבדק גם ש־`calibration_ui.py` (Ingestion בזמן Calibration) ו־`gaze_features.from_observation` (Live) קוראים בדיוק אותם שדות גולמיים (`left_eye.iris_in_eye`/`iris_in_lids_y`) ומפעילים את אותה פונקציית `_build()` — נשלל Sign/Axis-swap/Feature-mismatch/Ordering bug (השערות 1–3).

נמצא קונפאונד אמיתי: ב־Dataset הנוכחי `head_pitch_deg` מתואם `r=0.72` עם `left_iris_y` (ו־`0.63` עם `right_iris_y`), כי מסך Ultrawide 4096px מניע תנועת ראש טבעית בזמן הבטה ל־Targets עליונים/תחתונים. פרופיל ה־Y הקיים `AXIS_IRIS` כלל את Pitch, ו־Leave-One-Target-Out (שכבר ממומש ב־`cross_validate`) הראה שהמקדם על Pitch תלוי בצימוד הזה: הסרתו הורידה P95 Y מנורמל מ־`0.3999` ל־`0.2684` (ירידה של כ־33% בשגיאת המקרה הגרוע) על אותו Dataset, כמעט ללא פגיעה ב־Median (`0.1390→0.1421`). הוספת אותה הסרה ל־`AXIS_IRIS_LIDS` החמירה שם P95 (‏`0.5652→0.5810`), ולכן הפרופילים נשארו שונים במכוון — כל אחד לפי הראיה שלו, לא לשם אחידות.

Polynomial Ridge (`AXIS_IRIS_LIDS`) אושר שוב כ־Overfit קטלני על אותו Dataset: P95 `3949.7px` ב־LOO, ולכן `advanced_improves_conservatively` ממשיך לדחות אותו והוא נשאר Benchmark בלבד — לא נכנס ל־Live validation candidates (מומש כבר, לא שונה).

תיקון קוד: `_Y_AXIS_IRIS_FEATURE_INDICES` ב־`src/gazelink/gaze_model.py` שונה מ־`(left_iris_y, right_iris_y, head_pitch_deg)` ל־`(left_iris_y, right_iris_y)` בלבד, עם הערה מתעדת את המדידה. זהו שינוי שורה אחת, מכוסה ב־Suite הקיים ללא שינוי בבדיקות. `_Y_AXIS_IRIS_LIDS_FEATURE_INDICES` לא שונה.

Target 0 (‏UP_LEFT) נשאר החריג הגרוע ביותר בכל וריאציה (LOO error עד `0.42–0.59` מנורמל) — יש לו yaw `31°`/pitch `-33°` קיצוניים ביחס לשאר, כנראה Extrapolation לפינה שדורשת סיבוב ראש חריג ב־Grid האופקי של 15%/85%. זו נשארת סיכון פתוח לתיעוד, לא תוקן ולא שונו נקודות ה־Edge inset.

QA אוטומטי: Suite מלא `315 passed, 2 deselected` (ללא רגרסיה). לא בוצע Run מצלמה חדש ולא נוצר Calibration חדש — כל המדידות לעיל הן Replay דטרמיניסטי על ה־Dataset הקיים `dataset_20260902T114309350044Z.json` דרך `CalibrationEngine().train()`/`cross_validate()` בפועל, ללא הפעלת מצלמה או OS input. `latest_model.json` עדיין לא קיים ולא נוצר על ידי שינוי זה; `pending_validation.json` הקיים לא נגע בו.

אימות חי לאחר התיקון — 2026-09-02

ה־Run החי הראשון אחרי התיקון (`validation_comparison_20260902T122317462058Z`) בדק בטעות candidate ישן: `pending_validation.json` הצביע עדיין על `LINEAR/FULL`/`POLYNOMIAL_RIDGE/FULL` (10 מקדמים, כל 9 ה־Features הגולמיים) מ־11:43, לפני שהתיקון נכתב. אומת ישירות מול `candidate_model_20260902T114309385971Z.json` (`profile` חסר בקובץ = FULL). לכן שוחזר מחדש: `CalibrationEngine().train()` הורץ שוב על `dataset_20260902T114309350044Z.json` הקיים (ללא מצלמה) עם הקוד המתוקן, ו־`CalibrationStore.save_pending_models()` נכתב מחדש עם `LINEAR/AXIS_IRIS` (`de4daae56dcf…`) ו־`LINEAR/AXIS_IRIS_LIDS` (`d21b5759a7c3…`). `latest_model.json` עדיין לא קיים.

Run שני (`validation_comparison_20260902T122629557205Z`) מדד את המודל הנכון וזו ההשוואה האמיתית מול ה־FULL הישן, על אותם Targets: CENTER ‏210px→110px, UP_CENTER ‏310px→218px, UP_RIGHT ‏266px→85px, RIGHT_CENTER ‏410px→77px (LEFT_CENTER נשאר דומה, ‏261px→259px). זהו שיפור אמיתי, שנמדד Live ולא רק ב־LOO. בהשוואת שני המועמדים החדשים, `AXIS_IRIS` (בלי Pitch) עדיף בבירור ב־3/5 Targets (UP_CENTER/UP_RIGHT/RIGHT_CENTER) ו־`AXIS_IRIS_LIDS` עדיף רק במעט ב־2 (CENTER/LEFT_CENTER); ה־P95 הגרוע ביותר בכל סט Targets נמוך יותר אצל `AXIS_IRIS` (‏299px מול ‏330px). הכלי עצמו הדפיס `Recommendation: none` כי אף מועמד אינו Dominant בכל Target — זו ההתנהגות השמרנית הנכונה ואינה תקועה.

`FEATURE_DRIFT_SUSPECTED` נורה ב־4/5 Targets גם ב־Run השני, לרוב על `head_yaw_deg`/`head_roll_deg`, עם z עד `8.25` דווקא ב־UP_CENTER — הנקודה עם השארית הגרועה ביותר. חשוב לנתח בזהירות: ה"Expected" של האבחון הזה הוא אינטרפולציה בין עוגני ה־Grid (0.05/0.5/0.95), ו־UP_CENTER/LEFT_CENTER/RIGHT_CENTER יושבים מחוץ ל־Grid בכוונה (M2-03 existing design) — לכן חלק מה־Drift עשוי לשקף שגיאת אינטרפולציה של ה־Reference עצמו ולא בהכרח סטיית ראש אמיתית של המשתמש. אין עדיין דרך להפריד בין השניים; זו לא נבדקה.

נשאר: אין המלצת קידום כרגע (נכון ותקין). הצעד הבא הוא להחליט אם `AXIS_IRIS` (העדיף כרגע ב־P95 ובמרבית ה־Targets) ראוי לקידום ידני מודע לאחר עוד Run חי אחד או שניים לאישוש היציבות, ו/או לחקור את מקור ה־Drift ב־UP_CENTER לפני כן. לא להריץ Calibration חדש — ה־Dataset עדיין תקין ומספיק.

Run שלישי לאישוש יציבות — 2026-09-02

`validation_comparison_20260902T122952740771Z` (אותם Candidates, ‏`de4daae56dcf`/`d21b5759a7c3`) מאשש שני דברים: (1) `AXIS_IRIS` ממשיך להיות עדיף באופן עקבי על `AXIS_IRIS_LIDS` ב־UP_CENTER/UP_RIGHT/RIGHT_CENTER על פני שני Runs נפרדים (UP_CENTER השתפר גם בפועל: ‏218px→106px, תחזית `y=0.21` מול Target `0.18`; UP_RIGHT ‏85px→98px median אך `y=0.16` כמעט מדויק). (2) `LEFT_CENTER` הוא כשל אמיתי ועקבי, בלתי תלוי במודל: שני המועמדים קיבלו P95 קטסטרופלי (`922px`/`924px`, לעומת `299px`/`273px` ב־Run הקודם), ו־`FEATURE_DRIFT_SUSPECTED` שם דיווח Feature שונה בכל Run (‏`head_yaw_deg` ב־Run הקודם, ‏`head_pitch_deg` כאן) — כלומר חוסר יציבות כללי בתנוחת הראש באזור הזה, לא קונפאונד יחיד וקבוע.

מסקנת ביניים: `AXIS_IRIS` הוא המועמד המועדף אם וכאשר תתקבל החלטת קידום, אך `LEFT_CENTER` (‏x=0.18 על מסך 4096px) נשאר אזור לא אמין בשני המודלים כאחד ומחייב בדיקה נפרדת — קרוב לוודאי תנוחת ישיבה/טווח תנועה א־סימטרי של המשתמש ולא באג קוד. לא בוצע שינוי קוד נוסף; אין קידום.

בדיקת השערת התנוחה — נשללה — `feature_check_20260902T123348515856Z` — 2026-09-02

הריצה המבוקרת מפריכה את ההשערה שכיוון LEFT גורם לחוסר יציבות ראש: `head_pose_p95_deg` בכיוון LEFT היה `0.60952°` — נמוך יותר מ־CENTER עצמו (`0.99848°`) ומתחת בהרבה לסף `5°`. כל חמשת הכיוונים היו יציבים (`stationary_p95` ‏`0.0135–0.0218`, מתחת לסף `0.08`). כלומר המשתמש כן מסוגל להחזיק ראש יציב כשמסתכל שמאלה; זו לא בעיית ישיבה/טווח תנועה כללית.

`Overall diagnostic pass: False` בריצה הזו רק בגלל RIGHT ‏(`NO_SEPARATION`, ‏combined_delta=`0.02638` מול סף `0.03`) — אות חלש חד־פעמי, לא קשור ל־LEFT ולא חוסם כלום (לא רצים Calibration חדש).

מסקנה מתוקנת: ה־P95 הקטסטרופלי ב־LEFT_CENTER בשני Runs של `--gaze-validation` (‏`922px`/`924px`) כנראה נובע מ־Sample size קטן (‏`sample_count=7` ל־Target) ולא מבעיה מבנית — עם 7 דגימות, P95 קרוב ל־Max ופריים בודד חריג (Blink/micro-saccade רגעי) יכול להטות אותו לגמרי בעוד ה־Median נשאר סביר (‏`259–288px`). אין עדות תומכת ל־Drift מבני אמיתי באזור הזה; ה־Median הוא המדד האמין יותר כרגע ל־n כה קטן. אין המלצה לשנות Sample count/Capture window בלי אישור המשתמש — זהו Placeholder קיים.

נשאר לפני DONE

ה־Preflight המבוקר האחרון עבר ב־Replay תחת ה־Gate הדו־עיני המאושר. אין צורך לחזור עליו לפני ה־Session הבא; אם התנהגות המצלמה/תנוחת המשתמש משתנה מהותית, מריצים אותו שוב לפני Calibration.

להריץ `python -m gazelink --guided-calibration` מול המצלמה עם הקוד החדש, להשלים Session ולוודא שנוצרים Dataset JSON ו־Candidate Model JSON ושמודפס Benchmark אמיתי של Linear מול Advanced. לאחר מכן להריץ `python -m gazelink --gaze-validation`, לבדוק במיוחד CENTER/UP_CENTER/UP_RIGHT, ולפרסם Candidate רק באמצעות `A` אחרי סקירת המדדים. `X` דוחה אותו ללא שינוי ב־Active model.

להריץ `python -m gazelink --gaze-check` ולוודא ויזואלית Raw X/Y חי, Screen geometry נכונה, Edge behavior ו־DPI על מסך ה־Ultrawide.

להפעיל את זרימת Correction האבחונית מול המשתמש: Capture על Target ידוע, Validation Target נפרד, Before/After אמיתי, Accept/Reject, Restart persistence, Undo/Reset ו־Disable שחוזר ל־Base.

למדוד ולדווח Median/P95/per-region Error ו־Prediction latency על Dataset ה־JSON האמיתי. הלוגים הישנים הם טקסט אנושי מעוגל ל־3 ספרות ואינם תחליף ל־Dataset lossless; לא נעשה מהם Benchmark מוצר מזויף.

לכוון מול נתונים אמיתיים את כל ה־Placeholders: Ridge lambda, Gaze max age, Correction radius/shrinkage/max offset/conflict threshold, 300–500ms capture window, sample count, stability thresholds, stabilization delay ו־Validation acceptance.

כיוון Filter לפי stationary/step-response חי, Validation targets מלאים, per-region/corner error ו־real-camera M2 QA נשארים ב־M2-04. Blink activation נשאר ב־M3-03 ו־Hands-free polished UX נשאר ב־M4.

Definition of Done

Observation חי מהמצלמה יכול להפוך ל־Screen X/Y אמיתי לאחר Calibration.

Legacy mapping

M2-T04, M2-T05, M2-T07

M2-04 — Gaze Stability + Validation + Demo

Status: 🟡 PARTIAL / IN PROGRESS — Filter candidates, reset policy, rolling jitter metrics ו־Raw/Filtered diagnostic overlay מומשו ואומתו אוטומטית; נדרש Benchmark חי, בחירת Filter מכוילת ו־Validation/Accuracy מלאים. בנוסף, כלי מדידה ל"דיוק אמיתי מול שינון" (ראו למטה) מכסים עכשיו את "Validation targets נפרדים", חלק מ"Median/P95 error" ואת "Gaze point overlay" מתוך הרשימה הבאה — קוד בלבד, טרם אומת מול מצלמה אמיתית

לבצע

Compare EMA / One Euro / Kalman.

Jitter benchmark.

Filter reset ב־Lost.

Filter reset ב־new calibration.

Validation targets נפרדים.

Median/P95 error.

Region/corner error.

Gaze point overlay.

Full M2 real-camera QA.

Definition of Done

אחרי Calibration מופיעה נקודה יציבה שעוקבת אחרי המקום שבו המשתמש מסתכל, עם Accuracy שנמדדה מספרית.

Legacy mapping

M2-T06, M2-T08, M2-T09

כלי מדידה — דיוק אמיתי מול שינון — בוצע אוטומטית, טרם אומת מול מצלמה אמיתית — 2026-09-03

המשתמש ביקש יום מדידה נפרד: לפני שיפור נוסף, לדעת אם הדיוק הנמדד בכיול (250px→50-150px) הוא למידה אמיתית או ששיטת ה־Validation הקיימת (`--gaze-validation`, 5 Targets קבועים) פשוט חופפת חלקית לרשת הכיול עצמה — `CENTER` הוא ממש Calibration target 4, ו־`LEFT_CENTER`/`RIGHT_CENTER` יושבים על שורת האמצע שלה. חוקי ברזל מפורשים מהמשתמש: אסור לגעת ב־Model/Mapping/Calibration logic; כל שינוי הוא תוספת; אין המצאת פורמט קבצים חדש.

נוספו ארבעה קבצים חדשים תחת `src/gazelink/`, אפס עריכה ב־`calibration.py`/`calibration_ui.py`/`gaze_model.py`/`gaze_features.py`/`gaze_correction.py`/`gaze_engine.py`/`live_validation.py` (אומת ב־`git diff --stat`):

`test_points.py` — `generate_test_targets()`: Targets רנדומליים עם Seed קבוע (`20260903`, ניתן לשינוי), מרחק מינימלי מכל אחת מ־9 נקודות הכיול האמיתיות (נגזר מ־`targets_for_screen_geometry()` עצמה, לא רשימה משוכפלת) ומכל Target אחר. ברירת מחדל 10 נקודות. אזור העבודה נפתח במפורש מעבר לטווח הכיול ב־X (0.12–0.88 מול 0.15–0.85 בפועל על מסך Ultrawide) כדי לבדוק גם Extrapolation, לא רק Interpolation; `is_extrapolated()`/`calibration_bounding_box()` מסמנים זאת.

`test_dataset.py` — `TestSample`/`TestPointResult`: שכפול Byte-identical של סכמת `CalibrationSample`/`dataset_*.json` עם N Targets במקום 9 בדיוק (`CalibrationSample.target_index` נעול ל־0..8). `TestSample` בנוי סביב Carrier אמיתי מסוג `CalibrationSample` כדי לעטוף Validation קיים ולקרוא ל־`from_calibration_sample()` המקורי — אין העתקת חישוב Features. `TestSampleRecorder` עוטף את `LiveValidationController` הקיים (ללא עריכה בו) ומסיק מה קרה אך ורק מה־View המוחזר; מסרב לכתוב שורות אם ספירתן לא תואמת את מה ש־Controller דיווח. נכתב ל־`.gazelink/test_points/`, ספרייה נפרדת מ־`.gazelink/calibration/`.

`prediction_overlay.py` — `compute_prediction_overlay_state()` (טהור, ניתן לבדיקה בלי Qt) + `PredictionOverlay` (Qt shell דק): מציג `✕` מג'נטה בזמן אמת לצד המטרה, ומוחק אותו מיידית ברגע שאין Sample — לא מותיר נקודה קפואה. כישלון גאומטרי (מודל שלא תואם את המסך החי) מוצג כטקסט מפורש, לא כנקודה ריקה שקטה. `load_overlay_model()` טוען מודל מפורש בלבד — בלי `--overlay-model` אין נקודת חיזוי בכלל, כפי שהוחלט מול המשתמש.

`test_window.py` — מסך `--gaze-test` חדש, מקביל ל־`validation_window.py`: איסוף דגימות עובד גם בלי מודל מאומן בכלל (Heartbeat מלאכותי שמניע רק את שעון היציבות של ה־Controller, לעולם לא נקרא על־ידי ה־Recorder), וגם עם `--overlay-model` אמיתי (אותה תוצאה מזינה גם את השעון וגם את הסמן). `Esc`/`R`, אפס OS input.

`calibration_window.py`/`app.py` נערכו בתוספת בלבד: `run_guided_calibration`/`_CalibrationWindow` מקבלים `overlay_model_path` אופציונלי (ברירת מחדל `None`, מתנהג בדיוק כמו קודם); ארבעה דגלים חדשים (`--gaze-test`, `--test-seed`, `--test-points`, `--overlay-model`).

`analyze.py` (שורש הריפו) — קורא בלבד, כותב רק ל־`history.csv`. `--model` חובה כדי שלא תתערבב השוואה בין שני מודלים שונים. מדפיס `CALIB_MEDIAN`/`TEST_MEDIAN`/`GAP` (חציון, לא ממוצע), ואז `VERDICT: PASS` רק אם `TEST_MEDIAN < 120px` וגם `GAP < 50px` — שני התנאים יחד, נעולים מראש ולא ניתנים לשינוי בזמן ריצה. `TEST_MEDIAN_INTERP_ONLY`/`TEST_MEDIAN_EXTRAP_ONLY` מדווחים כפירוק אבחוני בין נקודות בתוך/מחוץ לגריד הכיול, במפורש **לא** משתתפים בהכרעה (מכוסה בבדיקה ייעודית: חציון־פנימי נמוך לא הופך FAIL כללי ל־PASS). `--include-rejected` בונה עותק זמני עם `accepted=True` (`dataclasses.replace`) כדי להעביר גם דגימות שנדחו דרך אותה פונקציית Feature extraction בלי לעקוף אותה.

תועד ב־`docs/MEASUREMENT_DAY.md` (חדש) — לא נערך `README.md` הראשי.

QA אוטומטי: 365 בדיקות עוברות (350 קיימות + 15 `test_analyze.py` + קודמות ב־Stage 1/2), `ruff check .`, `ruff format --check .`, `mypy` (ללא ארגומנטים; `analyze.py` נוסף במפורש ל־`[tool.mypy].files` ב־`pyproject.toml` כך שנכלל גם בברירת המחדל) — כולם נקיים. הרצת Smoke ידנית של `analyze.py` על Fixtures סינתטיים (מודל קבוע החוזה נקודה יחידה) אימתה שהפלט תואם בדיוק את מה שמתועד ב־README, כולל אזהרת Geometry mismatch, ושקבצי הקלט לא נגעו בהם (mtime זהה) — נמחקו לאחר מכן.

נשאר: אין Run מצלמה אמיתי. `--gaze-test`/`--overlay-model` לא הופעלו מול Webcam אמיתי; כל האימות הוא דטרמיניסטי (Fixtures סינתטיים, מודל קבוע-חיזוי). ה־250px→50-150px שנמדד קודם טרם נבדק דרך הכלים החדשים — זו בדיוק המדידה שהמשתמש צריך לבצע בעצמו (הסכמה מפורשת: "אני עושה: מריץ את הכיול... אתה עושה: בונה כלים"). אין המלצה על סף PASS/FAIL אחר מזה שהמשתמש קבע מראש.

כלי מדידה — השוואה לספרייה חיצונית (predictions mode) — בוצע אוטומטית, טרם אומת מול Webcam/EyeGestures אמיתיים — 2026-09-03

המשך ישיר לכלי המדידה שלמעלה: המשתמש רץ ניסוי נפרד ומבודד לגמרי (venv/ריפו משלו תחת `~/experiments/eyegestures-test/`, לא קשור ל־gazelink) עם ספריית `eyeGestures` חיצונית (Third-party, רישיון GPL-3.0), וביקש דרך להשוות אותה למערכת שלנו על אותו סרגל בדיוק — בלי לגעת במודל/מיפוי/כיול שלנו ובלי לשלב את EyeGestures לתוך gazelink. חוקי ברזל מפורשים מהמשתמש: אין עריכת אלגוריתם/מיפוי/כיול קיימים; אין אינטגרציה של EyeGestures לתוך gazelink; אותן נקודות בדיוק, אותו Seed, אותו סרגל (מסך) — אחרת ההשוואה חסרת ערך.

`analyze.py` (שורש הריפו) הורחב, לא נכתב מחדש: דגל חדש `--predictions <file.json>` שמחליף את `--test`+`--model` יחד (`--calib` נשאר גם הוא לא רלוונטי במצב הזה — Parser מסרב לשלב `--predictions` עם כל אחד מהשלושה, ומחייב את כל השלושה יחד כשאין `--predictions`). קורא `PredictionSample`/`PredictionResult` חדשים (Target index + X/Y נורמליזציה מוכנה + Timestamp + Accepted, פלוס Targets ו-Screen geometry ברמת הקובץ — פורמט חדש, מוצהר במפורש, לא מומצא תוך כדי) ומחשב שגיאות דרך `compute_prediction_errors()` חדשה שמשתמשת באותה `_pixel_error()`/`is_extrapolated()` בדיוק כמו `compute_test_errors()` — בלי לקרוא ל־CalibrationModel ובלי Feature extraction בכלל (אין מודל שלנו לערב). הרנדור שותף: `_format_file_section()`/`_format_diagnostic_breakdown()` חדשים חולצו מתוך `format_report()` הקיים (רה־שימוש, לא כפילות) ומוזנים גם ל־`format_predictions_report()` החדש. במצב Predictions: `CALIB_MEDIAN`/`GAP` תמיד `n/a`, ואין `VERDICT` בכלל (לא FAIL, לא PASS — הסף הקבוע מעולם לא אומת מול מקור בלי Gap משלו); `history.csv` מקבל שורה עם `verdict=N/A` ו־`model_id=predictions:<שם קובץ>`. אפס עריכה בקוד הקיים של `compute_calibration_errors`/`compute_test_errors`/`compute_verdict`/`append_history` (אומת ב־Diff).

`export_test_targets.py` (שורש הריפו, קובץ חדש) — כלי Read-only נוסף מקביל ל־`analyze.py`: קורא ל־`generate_test_targets()` הקיים (אותה פונקציה בדיוק ש־`--gaze-test` עצמו קורא לה, לא העתק) עם ברירת המחדל של Seed/Count/Screen geometry שלה, וכותב JSON עם `targets`+`screen_geometry` בלבד — אותה צורה בדיוק שקובץ Predictions מצפה לה, כדי שניסוי חיצוני יוכל להרחיב אותו במקום לשכתב פורמט. `--width-px`/`--height-px` ברירת מחדל מיובאת ישירות מ־`analyze.SCREEN_WIDTH_PX/HEIGHT_PX` (מקור אמת יחיד, לא מספרים משוכפלים).

`eyegestures_capture.py` — נכתב **מחוץ לריפו הזה לגמרי**, תחת `~/experiments/eyegestures-test/` (venv הנפרד של EyeGestures), לפי דרישת המשתמש המפורשת. אפס Import של חבילת `gazelink`; קורא בלבד את `targets.json` (JSON גנרי, לא תלות בקוד) וכותב Predictions JSON בפורמט של סעיף analyze.py. משתמש אך ורק ב-API הציבורי של EyeGestures (`EyeGestures_v2`, `VideoCapture`) לפי המתכון המדויק מ־`examples/simple_example_v2.py` שלהם (Calibration map מסוג Meshgrid, `setClassicalImpact(2)`, `setFixation(1.0)`) — אין עריכת קוד שלהם, ואין העתקה/שינוי של הלוגיקה הפנימית שלהם. Ctrl+Q/סגירת חלון מבטלים בבטחה בלי לכתוב פלט חלקי.

QA אוטומטי (בתוך gazelink בלבד — eyegestures_capture.py לא ב-CI של gazelink ולא ניתן להריצו ב-venv שלו, Python/Deps שונים לגמרי): 376 בדיקות עוברות (365 קיימות + 8 `test_analyze.py` חדשות למצב Predictions + 3 `test_export_test_targets.py` חדשות), `ruff check .`, `ruff format --check .`, `mypy` (ללא ארגומנטים; `export_test_targets.py` נוסף גם הוא במפורש ל־`[tool.mypy].files`) — כולם נקיים. `export_test_targets.py` הורץ ידנית והושווה ל־Import ישיר של `generate_test_targets()` — Byte-identical. `eyegestures_capture.py` עבר רק Compile+Import smoke test בתוך ה-venv הנפרד שלו (`py_compile` + Import ללא הרצת `main()`) — **לא הורץ מול מצלמה אמיתית או מול EyeGestures החי**.

נשאר: כל זרימת EyeGestures החיה (Calibration שלהם, Capture על 10 הנקודות המוחזקות, כתיבת Predictions JSON, והרצת `analyze.py --predictions`) לא בוצעה בפועל — זו בדיוק החלוקה שהמשתמש ביקש ("אני ארוץ פיזית מול המצלמה... אתה לא צריך להעריך את זה בשבילי"). לפני הרצה: לוודא שה־`--width-px`/`--height-px` שהוזנו ל־`export_test_targets.py` (ברירת מחדל 4096x1152) תואמים בפועל למסך הפיזי ש־EyeGestures ירוץ עליו — אחרת "אותו סרגל" לא מתקיים. אין הרצה מול GPU/מצלמה שונה מזו שנבדקה בניסוי הבידוד הקודם.

מסלול חיזוי חלופי — `--engine eyegestures` — בוצע אוטומטית, טרם אומת מול מצלמה — 2026-09-03

⚠️ **חוב רישוי פתוח — GPL-3 — חוסם את M6.** `eyeGestures` מופץ תחת GPL-3.0 בעוד `pyproject.toml` מצהיר `license = "LicenseRef-Proprietary"`. יבוא בתהליך יוצר, לפי הפרשנות המקובלת, יצירה משולבת: **הפצת GAZELINK יחד עם הספרייה תחייב שחרור של GAZELINK כולו תחת GPL-3.** המשתמש אישר במפורש **שימוש פנימי בלבד** בשלב הזה (GPL הוא רישיון הפצה; שימוש שלא מופץ אינו מפעיל את החובות). M6 ב-`spec.MD` הוא "משתמש חדש יכול להתקין, לכייל ולהשתמש" — כלומר **הפצה** — ולכן M6 חסום עד רישיון מסחרי (`contact@eyegestures.com`, ה-README שלהם מאשר שקיים) או החלטה אחרת. לא ניתן ייעוץ משפטי; נדרש ייעוץ אמיתי לפני הפצה. הקלה מבנית: התלות היא **extra אופציונלי** (`[project.optional-dependencies].eyegestures`) עם יבוא עצל — סגור התלויות של התקנת ברירת המחדל נשאר נקי מ-GPL, והמסלול ה-native עובד גם בלי שהספרייה מותקנת בכלל.

חוקי ברזל מהמשתמש: אין מיזוג של הלוגיקה שלהם לתוך `gaze_model.py`/`gaze_features.py`/`calibration.py`; אין שינוי במנוע הקיים; זיהוי הקריצות נשאר שלנו בשני המסלולים; שני המסלולים מוציאים אותו טיפוס נקודה. **קבצים שלא נגעתי בהם**: `gaze_engine.py`, `gaze_model.py`, `gaze_features.py`, `calibration.py`, `calibration_ui.py`, `runtime.py`, `camera.py` (אומת ב-Diff).

⚠️ **חריגה מהצהרה קודמת: `vision.py` כן נערך.** בגרסה מוקדמת של הרשומה הזו נכתב שלא נגעתי בו — זה היה נכון אז והתברר כבלתי-אפשרי לשמר. שורה אחת: המודל נטען דרך `model_asset_buffer` (bytes) במקום `model_asset_path`. הנימוק תחת "התנגשות MediaPipe" למטה. זה שינוי בטעינת קובץ ולא בלוגיקה — אותו מודל, אותן אפשרויות, אותו פלט — ואומת שהמסלול ה-native ללא שינוי בביצועים (32ms/31FPS לפני ואחרי). המשתמש עודכן ואישר.

חוסם שנפתר לפני האינטגרציה: ה-venv הריץ `mediapipe 0.10.35`, שממנה **הוסר** ה-API הישן `mp.solutions` ש-EyeGestures תלויה בו, בעוד `vision.py` שלנו על ה-Tasks API החדש. אומת ש-`0.10.21` תומכת בשתי ה-APIs, בוצעה הורדה, ו**המנוע הקיים אומת ששרד**: 376 בדיקות עוברות (זהה ל-baseline), ruff/mypy נקיים, ובנוסף הורץ `FaceLandmarkerAdapter` האמיתי עם קובץ המודל האמיתי על פריים סינתטי (המודל נטען, `detect_for_video` רץ, החזיר `LOST`/`FACE_NOT_FOUND` כצפוי לתמונה שחורה, `close()` תקין). המשתמש הריץ `--guided-calibration` מול מצלמה חיה אחרי ההורדה והיא הושלמה — נכתבו `dataset_*.json`, שני `candidate_model_*` ו-`pending_validation.json`, כלומר חילוץ ה-Landmarks עובד מקצה לקצה. הסף נרשם ב-extra: `mediapipe>=0.10.21,<0.10.30`.

`gaze_predictor.py` (חדש) — התפר: `GazePredictor` Protocol שמקבל **גם** `FramePacket` וגם `VisionObservation` ומחזיר את `GazeEstimationResult` הקיים, כך ששום צרכן במורד הזרם לא יודע מי רץ. `NativeGazePredictor` הוא עטיפה דקה סביב `GazeEstimator` הבלתי-נגוע ומתעלם מה-Frame (הוא כבר נצרך במעלה הזרם). כל היבואים בקובץ הם `TYPE_CHECKING` בלבד — כדי ש-`app.py` יוכל לקרוא את שמות המנועים ל-Parser בלי לגרור את המנוע (ואת numpy) למסלול `--smoke`; אומת: `import gazelink.app` = 26ms, אפס מודולים כבדים, והבדיקה הקיימת `test_default_and_smoke_paths_never_import_a_camera_or_ui_stack` ממשיכה לעבור.

`eyegestures_engine.py` (חדש) — מתאם בלבד. `frame_to_rgb_array()` פונקציה טהורה שממירה `FramePacket.image` (bytes) ל-`ndarray` רציף ב-RGB (RGB ולא BGR — כך הדוגמה העובדת שלהם מזינה את `step()`), ומחזירה `None` — לא מערך חלקית-תקין — על פורמט לא נתמך/`image=None`/buffer שלא תואם לגאומטריה המוצהרת. `EyeGesturesGazePredictor` מחזיק מכונת מצבים `UNCALIBRATED→CALIBRATING→READY` וסופר נקודות כיול שהושלמו ע"י מעקב אחרי תזוזת `Cevent.point` — **בלי לגעת בשדות פרטיים** כמו `clb[ctx].fitted`. `build_calibration_map()` משתמש ב-`default_rng(seed)` ולא ב-`np.random.shuffle` הגלובלי של הדוגמה שלהם, כדי שסדר הכיול יהיה משוחזר בין ריצות.

שתי תכונות בטיחות שהמתאם קיים כדי לאכוף: (1) **מלכודת ה-[0,0]** — `Calibrator_v2.predict()` שלהם מחזיר `[0.0, 0.0]` לפני שהתאמן, כלומר הפינה השמאלית-עליונה של המסך, קואורדינטה שנראית לגמרי חוקית; המתאם מסרב לפלוט Sample כלשהו עד שרצף הכיול שלו הושלם. (2) **מדיניות הביטחון נשארת שלנו** — Sample נפלט רק כשה-`tracking_state` שלנו הוא `TRACKED`; אחרת מוחזרות הסיבות שלנו. בנוסף: `valid_for_control=False` תמיד; `confidence` נלקח מה-Observation שלנו (הם לא חושפים Confidence מכויל); נקודה מחוץ ל-0..1 מסומנת `OUT_OF_RANGE`+`CLAMPED_TO_SCREEN` כמו במנוע הקיים; נקודה לא-סופית (NaN/inf) נדחית כ-`ERROR`.

**תיקון עיצובי אחרי מדידה חיה — הזנת המנוע החיצוני מופרדת מפליטת הנקודה.** בגרסה הראשונה השער היה גם על הקריאה ל-`step()`: אם ה-tracker שלנו לא אישר את הפריים, הספרייה לא נקראה כלל. נמדד שזה **מרעיב את הכיול שלהם**: מנוע הראייה שלנו אישר רק **61 מתוך 333 פריימים (18%)**, ומכיוון שהם דורשים 30+ דגימות לכל נקודת כיול (`isReadyToMove`), הכיול התקדם **נקודה אחת ב-12 שניות** — כ-5 דקות לסבב מלא, מה שנראה למשתמש כמסך תקוע. עכשיו הספרייה מקבלת **כל** פריים (גם כאלה שהמדיניות שלנו פוסלת, וגם `observation=None`), והשער נשאר במקום היחיד שבו הוא באמת שער בטיחות — פליטת ה-`GazeSample`. הזנת פריים אינה פלט. שלוש בדיקות חדשות מקבעות את ההפרדה הזו.

`app.py` — `--engine {native|eyegestures}`, ברירת מחדל `native`. שילוב `--engine eyegestures` עם דגל שאינו `--gaze-check` נכשל ב-`parser.error` (יציאה 2) במקום להריץ בשקט את המנוע ה-native תחת דגל שהמשתמש חשב שנכנס לתוקף. `gaze_window.py` — `run_gaze_check(..., engine=...)` בונה את ה-Predictor המתאים; במסלול החיצוני אין מודל שלנו ולכן `CorrectionDiagnosticSession` הוא `None`, מקשי ה-Correction (K/1-9/A/X/U/R/D) הופכים אינרטיים במקום לעבוד חלקית, ומצויר יעד הכיול של הספרייה עם התקדמות `N/total`. היקף מכוון: **רק `--gaze-check`**. `--guided-calibration`/`--gaze-validation`/`--gaze-test` קשורים לסמנטיקת האימון/ולידציה של המודל *שלנו* ומנוע זר שם יהיה מטעה.

**באג אמיתי שנתפס בהרצת Smoke מול הספרייה האמיתית — ההנחה המקורית שלי הייתה שגויה.** הנחתי ש-EyeGestures מחזירה `(None, None)` כשאין פנים בפריים. זה נכון רק בחלק מהמסלולים: כש-MediaPipe מחזירה תוצאה שבה `multi_face_landmarks is None`, `face.py:67` שלהם עושה `None[0]` ו**זורק `TypeError`**. ה-`process()` שלהם עוטף את זה ב-try/except שמוער כהערה, כך ששום דבר במעלה הזרם לא מכיל את זה. מכיוון שזה בדיוק המצב הרגיל **לפני שהמשתמש מתיישב מול המצלמה**, החלון היה קורס על הפריים הראשון. התיקון הוחל **במתאם ולא ב-site-packages** (patch שם אינו משוחזר ונמחק בכל reinstall): `predict()` עוטף את הקריאה ל-`step()` בלבד ומתרגם חריגה ל-`FACE_NOT_FOUND` — גבול ההכלה הוא בדיוק תפקידו של המתאם. נוספו שתי בדיקות (`_ExplodingGestures`) שמקבעות גם את ההכלה וגם שאין הצטברות מצב על כשל חוזר. שווה לשקול לפתוח Issue אצלם.

QA אוטומטי: **412 בדיקות עוברות** (376 קודמות + 25 `test_eyegestures_engine.py` + 5 `test_gaze_predictor.py` + 4 CLI ב-`test_smoke.py`), `ruff check .`, `ruff format --check .`, `mypy` — כולם נקיים, גם כשה-extra מותקן וגם בלעדיו. הספרייה החיצונית מוחלפת בבדיקות בכפיל מוזרק בצורת ה-API שלה; אומת בפועל ע"י חסימת המודול ב-`sys.meta_path` ש-30 הבדיקות של שני הקבצים החדשים עוברות **כשהספרייה בלתי-זמינה לחלוטין**. קוד ה-ignore של mypy הועבר מ-inline ל-`[[tool.mypy.overrides]]` כי הוא תלוי סביבה (`import-not-found` בלי ה-extra מול `import-untyped` איתו). בדיקה אחת תפסה הנחה שגויה שלי לגבי תזמון דגל ה-`calibrate` (הדגל נבחר לפני הקריאה, ולכן הפריים שמשלים את הכיול עדיין מבקש כיול) — התוקנה הציפייה בבדיקה, לא הקוד.

אומת מול הספרייה האמיתית (בלי מצלמה, בלי GUI): בנייה אמיתית של `EyeGestures_v2`, קבלת מפת הכיול שלי, `setClassicalImpact`/`setFixation`, ו-`step()` אמיתי עם פריימים סינתטיים ב-BGR24 וב-RGB24 — כולם עברו, כולל הכלת הקריסה. אומת גם שמסלול ה-not-TRACKED לא נוגע בספרייה בכלל.

**שני חסמים שהתגלו רק בהרצה חיה, ושניהם נפתרו:**

1. **התנגשות MediaPipe (הרסנית ושקטה).** EyeGestures משתמשת ב-API הישן `mp.solutions`, ועצם הייבוא שלו מגדיר **resource root גלובלי** בתהליך שמצביע על חבילת MediaPipe, כדי שהגרפים שלה יפתרו נתיבים יחסיים. מאותו רגע ה-Tasks API — שבו `vision.py` שלנו משתמש — פותר גם נתיבים **מוחלטים** יחסית לשורש הזה ומייצר `site-packages/C:\Users\...\face_landmarker.task`. כל בניית `FaceLandmarker` נכשלה, `vision.py` בלע את החריגה והחזיר `ERROR` — כלומר **בחירת המנוע החיצוני הרגה בשקט את זיהוי הפנים של gazelink** (נמדד: 156/156 פריימים `ERROR`). איפוס ה-root **אינו** פתרון: הוא מתקן אותנו ושובר את הנתיבים היחסיים שלהם, וכישלונם נבלע ב-try/except שלהם — ניסיתי, ובדיקת האימות שלי פירשה בטעות שגיאה מוכלת כהצלחה. הפתרון הנכון הוא בצד שלנו: טעינת המודל מ-bytes, שחסינה לכל resource root. אומת ללא סינון פלט: אפס שגיאות טעינת משאבים מהגרף שלהם, ושני המנועים מעבדים פריימים באותו תהליך.

2. **קיפאון החלון.** Qt חד-חוטי; כשה-slot של הטיימר ארוך מהמרווח, אירועי ציור לא מקבלים זמן והחלון מפסיק להתרענן — נראה תקוע למרות שהקוד רץ. המנוע שלנו לבדו עולה ~32ms מול טיימר ברירת מחדל של 16ms (כבר חריגה פי 2), והוספת ה-FaceMesh שלהם (~15ms) הביאה ל-~47ms. במסלול החיצוני בלבד הטיימר הואט ל-`EXTERNAL_ENGINE_TIMER_INTERVAL_MS = 66` (~15 FPS, ~19ms פנויים לציור). אומת מול מצלמה על ידי המשתמש — הכיול והמעקב עובדים.

**ממצא נלווה שאינו קשור למנוע החיצוני ושווה טיפול נפרד:** מנוע הראייה שלנו אישר רק **18% מהפריימים** (61/333) בהרצה חיה. `display_eligible` דורש **אפס** reason codes, כך שכל מצמוץ או סטיית ראש קלה פוסלים פריים שלם. זה מאט גם את הכיול והוולידציה של gazelink עצמו, לא רק את המנוע החיצוני.

**אומת מול מצלמה חיה על ידי המשתמש**: `--gaze-check --engine eyegestures` עולה, מריץ את הכיול של הספרייה, ועובד. `--guided-calibration` ושאר המסלולים ה-native אומתו ללא רגרסיה (32ms/31FPS לפני ואחרי כל שינוי).

**מלכודת תפעולית שעלתה שלוש פעמים במהלך העבודה ושווה לזכור**: תהליך אחר שמחזיק את המצלמה גורם ל-`CameraError`, והחלון אז **מכבה את הטיימר** ומציג טקסט שגיאה קטן בלבד — מה שנראה זהה לחלוטין ל"החלון תקוע/ריק". הופיע בגלל (א) תהליך `eyegestures_capture.py` תקוע שלא סגר את המצלמה (תוקן שם ב-`finally: cap.close()`), ו-(ב) `tobi/tools/sweep_geometry.py` של המשתמש שרץ במקביל. לפני אבחון של "החלון לא עובד" — לבדוק `Get-Process python` קודם.

נשאר: לא נמדדה עלות CPU/FPS מדויקת של שתי הרצות FaceMesh לאורך זמן; ה-threads-per-frame שלהם נבדקו בסימולציה קצרה בלבד; ולא נמדדה **דיוק** המנוע החיצוני מתוך gazelink (רק דרך הכלי העצמאי, שנתן ~361-392px אחרי סינון זמן-נסיעה מול ~438px חציון של המנוע הקיים).

M3 — Control: העיניים מחליפות את העכבר

M3-01 — Safety + Windows Input

Status: ⬜ NOT STARTED

לבצע

SafetyController.

FakeInputAdapter.

RealWindowsInputAdapter.

Explicit opt-in.

Fail-closed states.

Release-all.

Definition of Done

אין OS action במצב לא בטוח, וניתן להפעיל Windows input רק במפורש.

Legacy mapping

M3-T01 עד M3-T03

M3-02 — Cursor Control

Status: ⬜ NOT STARTED

לבצע

Gaze → Cursor.

Screen bounds.

Sensitivity.

Low-confidence freeze.

Simulation mode.

Control latency measurement.

Definition of Done

הסמן זז לפי המבט.

Legacy mapping

M3-T04, חלק מ־M3-T10

M3-03 — Blink/Wink/Dwell Gesture Engine

Status: ⬜ NOT STARTED

כבר יש חומר גלם

Eye openness מ־M1.

Left/right eye data.

Monotonic timing infrastructure.

לבצע

Natural blink detector.

Left wink.

Right wink.

Intentional hold.

Duration thresholds.

Cooldown.

Double gesture.

Dwell.

Pause/Emergency.

Gesture → action mapping.

Integration עתידי עם M2-03B: Intentional Gesture יכול להפעיל Correction Mode, לבחור אזור ב־Auto-scan ולאשר Retry/Cancel/Accept. החיבור ייעשה רק לאחר שהמנוע מוכיח Natural blink separation, Cooldown ו־one event per gesture; Pause/Emergency נשארים בעלי עדיפות.

Definition of Done

המערכת יודעת להבדיל בין מצמוץ טבעי לבין פעולה מכוונת ומפיקה Event אחד נכון לכל מחווה.

Legacy mapping

M3-T05 עד M3-T08

M3-04 — Mouse Actions + POC

Status: ⬜ NOT STARTED

לבצע

Left click.

Right click.

Double click.

Dwell select.

Drag & Drop.

Pause/Resume.

HUD.

False activation benchmark.

End-to-end POC.

Demo

Look at Chrome → Cursor moves → Wink/Dwell → Chrome opens

Definition of Done

שליטה בסיסית ב־Windows עובדת ללא ידיים.

Legacy mapping

M3-T06 עד M3-T12

M4 — Usability: שימוש אמיתי במחשב

M4-01 — Targeting UX

Status: ⬜ NOT STARTED

Smart snapping.

Target expansion.

Zoom lens.

Gaze cursor.

Visual/audio feedback.

Scroll zones.

Personal dwell/sensitivity.

Hands-free Quick Regional Correction UX מעל מנוע M2-03B: Targets ידועים, Feedback חזותי/קולי, Target sizes, Contrast, Fatigue, Retry/Cancel/Undo, עברית RTL ואנגלית. Voice יכול להיות Shortcut אופציונלי בלבד ולא תלות להפעלה עצמאית.

Legacy mapping

M4-T01 עד M4-T05

M4-02 — On-Screen Keyboard

Status: ⬜ NOT STARTED

Keyboard shell.

Safe text injection.

Hebrew RTL.

English.

Blink/Dwell selection.

Word prediction.

Common phrases.

Legacy mapping

M4-T06 עד M4-T08

M4-03 — Profiles + Settings

Status: ⬜ NOT STARTED

Profile schema.

Calibration persistence.

Gesture thresholds.

Smoothing.

Dwell.

Language.

Profile selection.

Basic therapist settings.

Legacy mapping

M4-T09, M4-T10

M4-04 — Alpha QA

Status: ⬜ NOT STARTED

Demo

Browser → Scroll → Type → Send

לבצע

Endurance.

Fatigue.

Task completion.

Errors / false actions.

Full Hebrew/English demo.

Legacy mapping

M4-T11, M4-T12

M5 — Platform: מנוע קלט לאפליקציות

M5-01 — Public Contracts + Security

Status: ⬜ NOT STARTED

Threat model.

Consent/auth.

API versioning.

Public event schemas.

No raw frames/landmarks.

Legacy mapping

M5-T01, M5-T02

M5-02 — Event Bus + WebSocket/REST

Status: ⬜ NOT STARTED

Local event bus.

Slow-client isolation.

WebSocket.

REST.

Enter/Leave/Select semantics.

Legacy mapping

M5-T03 עד M5-T06

M5-03 — SDK + Browser

Status: ⬜ NOT STARTED

SDK.

Sample app.

Browser integration.

Legacy mapping

M5-T07, M5-T08

M5-04 — PANDY / LAXI + Beta QA

Status: ⛔ BLOCKED PARTIALLY

PANDY contract/access.

LAXI contract/access.

Integration אחת לפחות.

Beta demo.

Performance/load/security QA.

Legacy mapping

M5-T09 עד M5-T11

M6 — Product: V1

M6-01 — Accuracy / Reliability / Performance

Status: ⬜ NOT STARTED

Hardware matrix.

Accuracy hardening.

Calibration refinement.

Lighting / glasses / head movement / tremor.

Performance.

Soak tests.

Legacy mapping

M6-T01 עד M6-T04

M6-02 — Installation + Onboarding

Status: ⬜ NOT STARTED

Environment check.

Setup Wizard.

Guided calibration product flow.

Installer.

Auto-start.

Update/rollback/uninstall.

Legacy mapping

M6-T05 עד M6-T07

M6-03 — Profiles / Security / Therapist

Status: ⬜ NOT STARTED

Multiple profiles.

Migration/backup.

Encryption.

Data lifecycle.

User/therapist permissions.

Therapist console.

Safe fail/crash recovery.

Legacy mapping

M6-T08 עד M6-T11

M6-04 — Release QA + Pilot

Status: ⬜ / ⛔ PILOT BLOCKED UNTIL PRODUCT DECISIONS

Full release verification.

Hardware regression.

Pilot protocol.

Consent.

Pilot.

V1 Go/No-Go.

Legacy mapping

M6-T12 עד M6-T14

ההחלטות הפתוחות

D-01 — Reference Webcam רשמית.

D-02 — Stack.

D-03 — Single screen through M4.

D-04 — Default Gesture → Action mapping.

D-05 — Pause/Emergency gesture.

D-06 — Product accuracy / false activation targets.

D-07 — Therapist permissions / profile ownership.

D-08 — PANDY/LAXI contracts.

D-09 — Telemetry/support diagnostics policy.

D-10 — Pilot population / consent / success metrics.

הצעד הבא

M2-02 Stage A (שמירת Dataset אמיתי) הושלם ואומת מול מצלמה אמיתית — ראו M2-02 למעלה.

M2-02 Stage B (Sample Quality Policy) הושלם ברמת הקוד ואומת אוטומטית (236 tests, ruff/format/mypy/build נקיים, 26/26 Scenario QA) — ראו M2-02 למעלה. ה-Dataset שנאסף מעתה מסונן מ-Samples ישנים, מעיניים שאינן נראות בפועל, מזוויות ראש קיצוניות ומהצצות חריגות בתוך אותו יעד, וכל דחייה נושאת Reason code מקבוצה סגורה.

נשאר לפני שאפשר לסמן את M2-02 כ-DONE מלא:

הרצת Session אמיתי מול מצלמה עם המדיניות החדשה, ובדיקה בלוג ש-reason_counts() משקף דחיות אמיתיות והגיוניות (ובפרט: שאין Over-rejection שמקשה על סיום הכיול).

כיול מבוסס-מדידה של חמשת הספים החדשים — כולם Placeholders (ראו האזהרה ב-M2-02).

העבודה המרכזית הבאה היא Run אמיתי של `--gaze-feature-check` וניתוח ההפרדה/היציבות של CENTER/LEFT/RIGHT/UP/DOWN. אין להריץ Calibration נוסף לפני שה־Preflight מראה אם שתי העיניים נעות בעקביות או מצביע במפורש על `NO_SEPARATION`/`INVERTED`/`EYES_DISAGREE`. לאחר תיקון מקור ה־Feature חוזרים ל־Calibration עם Stabilization ושער Promotion, ואז ל־Raw מול Filtered. Foundations של M2-03A/M2-03B ושל M2-04 Filtering קיימים, אך Accuracy/Jitter חיים עדיין לא אומתו.

Feature schema 3 — vertical fallback diagnostic — בוצע אוטומטית, ממתין לאימות מצלמה — 2026-09-02

בדיקת ה־Feature החיה האחרונה (`feature_check_20260902T111823425241Z`) הראתה אות יציב אך אין הפרדת UP מציר פינות העין: combined `-0.00541`, נמוך מסף `0.015`. בהתאם אושר ונוסף `iris_in_lids_y` — מיקום הקשתית היחסי לפתיחת העפעפיים — כ־Feature 5/6 חדש, לצד ציר הפינות הקיים ולא במקומו. `FEATURE_SCHEMA_VERSION` עלה מ־2 ל־3; Dataset/Model/Correction ישנים נדחים במפורש כדי שלא יאומנו או ייטענו תחת משמעות Features חדשה.

`--gaze-feature-check` מודד ומדווח כעת גם `vertical_lid` ואת מקור ההכרעה האנכית: `CORNER_AXIS` נשאר ברירת־המחדל; `LID_RELATIVE` נבחר רק אם ציר הפינות נכשל בהפרדה והמדד החדש עובר את אותו Gate דו־עיני. לכן הבדיקה אינה מסתירה כשל באמצעות מעבר שקט לאות אחר. ה־Guided calibration וה־Gaze model נשענים על Schema 3 החדש בלבד.

קבצים ששונו: `domain.py`, `features.py`, `calibration.py`, `calibration_ui.py`, `gaze_features.py`, `feature_check.py`, `feature_check_window.py`, בדיקות ממוקדות והמסמכים האלה. QA אוטומטי: `312 passed, 2 deselected`, `ruff check`, `ruff format --check`, `mypy src tests`, `python -m gazelink --help` ו־`python -m build` עברו. נדרש כעת Run מצלמה אחד: `python -m gazelink --gaze-feature-check`; רק אם כל הכיוונים יציבים וכל ארבעת כיווני ההפרדה מדווחים `PASS` (ובפרט UP עם `vertical_source=LID_RELATIVE` או `CORNER_AXIS`) ממשיכים ל־`--guided-calibration`. אחרת עוצרים לפני אימון ומנתחים את הדוח החדש. לא הופעל OS input.

M2-03A ו־M2-03B מומשו ואומתו אוטומטית ב־2026-09-02. מנגנון איסוף מתוזמן, Model promotion gate ו־M2-04 Filter foundation נוספו לאחר כשל ה־Session החי. כעת נדרש Session חדש מול המצלמה; רק אם הוא מפרסם `latest_model.json` ממשיכים ל־Raw/Filtered ולזרימת Correction האבחונית Before/After. Blink activation נשאר תלוי ב־M3-03 ו־Hands-free product UX נשאר ב־M4.

הסיבה:

קיים עכשיו Dataset אמיתי (27 accepted + 11 rejected מ-Session חי) שגם עבר מדיניות איכות, כך שהמודל לא ילמד מרעש.

את הכיול הסופי של ספי האיכות עצמם קל יותר לבצע אחרי M2-03/M2-04, כשאפשר למדוד את השפעת כל סף על שגיאת הכיול בפועל ולא רק על ספירת Samples.

אחרי זה:

M2-03A Base Gaze Mapping → M2-03B Local Correction Foundation → M2-04 Gaze Demo → M3 Control

כלל עבודה ל־Claude Code

לפני כל Work Package:

קרא ROADMAP.md.

קרא את הסעיף המקביל ב־TECHNICAL_SPEC.md.

קרא את ה־Work Package כאן.

הצג בקצרה:

מה כבר קיים.

מה חסר.

מה היכולת המוצרית שנשיג בסוף העבודה.

אל תציג דרישות קיימות באפיון כרעיונות חדשים.

אל תסמן DONE על סמך קוד בלבד כאשר נדרש Camera/manual validation.
