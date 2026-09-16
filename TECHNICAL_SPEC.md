GAZELINK — אפיון טכני: ארכיטקטורה מודולרית

גרסה 2.0 | 15 בספטמבר 2026 | יעד מחייב למימוש הדרגתי

מסמך זה מחליף את האפיון הטכני המצורף מ־2026-09-01. הוא מגדיר מבנה יעד והוראות מימוש; אינו טוען שהריפקטור כבר בוצע. הבסיס: האפיון הטכני המקורי, אפיון המוצר GAZELINK v1.0 וסקירת הארכיטקטורה שסופקה בשיחה. הקוד עצמו לא היה זמין בעת עריכת גרסה זו. על Claude לאמת את ממצאי הסקירה מול ה־checkout הנוכחי לפני שינוי.

0. היררכיית מסמכים והחלטת ביצוע

ROADMAP.md: מטרות מוצר, Scope, Demo ו־Definition of Done לכל Milestone.

TECHNICAL_SPEC.md: מסמך זה — ארכיטקטורה, חוזים, בטיחות, פרטיות והחלטות טכניות.

TASKS.md: משימות, ראיות וסטטוס בפועל; אינו תחליף לאפיון.

docs/decisions/: החלטות טכניות והיסטוריית החלפתן.

הערת מאגר (15.9): במאגר זה מסמך המוצר נקרא spec.MD ומשמש בתפקיד ROADMAP.md. ההחלטה על מבנה הליבה בפועל מתועדת ב־docs/decisions/0002-modular-live-architecture.md.

המשתמש אישר לתעדף כעת ארכיטקטורה מודולרית ולבצע את השינויים הדרושים בקוד. האישור כולל פירוק רכיבים, הזרקת תלויות, איחוד לוגיקה שקולה, תיקוני lifecycle ובדיקות. אין צורך לבקש שוב אישור על כל שלב טכני שגרתי. אין בכך אישור לשנות מיפויי מחוות, ספי כיול, מודלי חיזוי, ברירות מחדל מוצריות או טכנולוגיית UI ללא צורך נפרד ומנומק.

החלטת הארכיטקטורה: לעטוף את המסלול החי הקיים ב־Adapters ולהעבירו לליבה מודולרית. אין להחליף מנוע או UI רק כדי להתאים לטבלת טכנולוגיות ישנה. ADR קיים יישמר כהיסטוריה ויסומן כמוחלף בחלקים הרלוונטיים. סתירה מוצרית אמיתית תתועד בנפרד; ממשיכים בעבודה שאינה תלויה בה.

1. מטרת המערכת

GAZELINK היא אפליקציית Desktop מקומית לשליטה ב־Windows באמצעות העיניים. Gaze קובע מיקום; Blink/Wink/Dwell קובעים כוונה ופעולה. הכיול, התיקון והשליטה מיועדים לתפעול ללא ידיים. עיבוד הווידאו מקומי בלבד.

מטרת השינוי הנוכחי: לאפשר פיתוח ותחזוקה בטוחים, בדיקה ללא מצלמה והחלפת תשתיות בלי לשכתב את לוגיקת המוצר. שינוי ארכיטקטורה אינו הבטחה לשיפור דיוק המבט.

2. מצב קיים והבחנה מדרישות יעד

הסטטוס מ־2026-09-01 מתאר שלב היסטורי ואינו בסיס לקביעה מה חסר כיום. לפי סקירת הקוד שסופקה:

קיימים מסלול מבט חי, כיול ומודלים, סינון, מחוות, Dwell, תפריט, גלילה ומקלדת.

מחלקות הליבה הקיימות כוללות הפרדת מצב ובדיקות שיש לשמר.

run_live מרכז אחריות רבה, ו־gf_record משמש גם כתשתית למערכת החיה.

המסלול שנבדק משתמש ב־gazefollower וב־pygame, בעוד שהאפיון הישן מתאר OpenCV/MediaPipe/PySide6 ישירים.

נמצאו בסקירה סיכוני שחרור קלט ומשאבים, כפילויות בעיבוד פריימים ותלות במבני הספרייה החיצונית.

Claude יתעד ב־TASKS מה אומת בקוד, מה נבדק אוטומטית ומה דורש חומרה. אין להסיק מהסקירה שכל יכולות המוצר גמורות, או שהמסלול הישן נמחק. הגרסאות והכניסות הפעילות ייקבעו לפי קוד ו־lockfiles, לא לפי זיכרון.

3. טכנולוגיות וגבולות ההחלפה

תחום

החלטה לריפקטור הנוכחי

Python ותלויות

לשמר את הגרסאות העובדות והנעולות בפרויקט; אין שדרוג סביבתי כחלק מפירוק קוד

מצלמה ומנוע חיצוני

המסלול הקיים מאחורי TrackingSource; GazeFollowerSource מרכז את החיבור ל־gazefollower

OpenCV / MediaPipe ישירים

אם יש מסלול פעיל נוסף, לשמרו כ־Adapter עצמאי; אין חובה להקים מחדש מסלול שאינו בשימוש

חיזוי וסינון

לשמר מודלים, סדר features, normalization, דיוק מספרי, presets ואלגוריתמים קיימים

ממשק

pygame ו/או Qt הקיימים מאחורי גבול תצוגה; אין מיגרציית UI בשלב זה

Windows input

Adapters קיימים מאחורי Ports; זהו המקום היחיד המורשה לפנות ל־OS

שמירה

פורמטים קיימים ותאימות למודלים/פרופילים; DB חדש אינו נדרש לריפקטור

Local API / SDK

הרחבות דרך אותם חוזים בעת ה־Milestone שלהן

IR ופלטפורמות נוספות

הגבולות מאפשרים Adapter עתידי; המימוש אינו חלק מהמשימה הנוכחית

4. ארכיטקטורה מודולרית מחייבת

4.1 כללי מבנה

מחלקות מחזיקות אחריות ומצב בעלי משמעות. חישובים טהורים נשארים פונקציות.

העדפת composition. אין היררכיות ירושה עמוקות, service locator או framework חדש להזרקת תלויות.

State מפורש ובעלים יחיד. אין מצב עסקי משתנה במילונים משותפים, globals או closures של פונקציית ההפעלה.

Protocol בגבול תשתית שנדרש לו מימוש אמיתי ומדומה: מקור מעקב, שעון, תצוגה וקלט. אין צורך בממשק לכל מחלקה.

ה־UI מציג מצב ומפיק כוונות; אינו מנבא מבט, מזהה קריצה או פולט קלט OS ישירות.

כלים וניסויים מייבאים את הליבה. הליבה אינה מייבאת CLI, כלי הקלטה, fitting או ניסוי.

אין גישה ל־face_info/gaze_info מחוץ ל־Adapter. נתון חסר מיוצג במפורש, לא כערך אפס מומצא.

לא מוסיפים שכבות רק להקטנת מספר שורות. כל רכיב חייב בעלות ברורה, ממשק קטן ובדיקה עצמאית רלוונטית.

4.2 רכיבים ואחריות

רכיב

אחריות בלעדית

גבול שאסור לחצות

GazeFollowerSource

פתיחה, warm-up, תרגום תוצאות הספרייה, פרסום observations וסגירה

אין החלטת קליק או UI

ReplaySource

שחזור features/timestamps מוקלטים לבדיקות

אין פתיחת מצלמה או קלט אמיתי

FramePipeline

קבלת observation, gate, prediction, filter ונתוני מחוות לפי המסלול הקיים

אין ציור או שליחת פעולות OS

CalibrationSession / שירותי כיול

איסוף, איכות, dataset ותוצר כיול; שימוש בליבת חיזוי משותפת

אין תלות בכלי CLI

Preflight

איסוף ואבחון לפני כיול לפי policy מפורש

אינו מחליף gate בטיחות בזמן שליטה

SafetyController

הרשאות לפי activation, freshness, tracking, calibration, pause/error ומצב פעולה

אינו מייצר החלטת מבט או מיפוי מחווה

InteractionController

מכונת מצבי UI ותזמור מחוות, Dwell, תפריט, גלילה ומקלדת

אין קריאות תשתית ישירות

ActionExecutor

בדיקת הרשאה עדכנית והעברת Actions ל־Ports

אין לוגיקת מחווה נוספת או עקיפת armed

Input Adapters

קלט Windows, מעקב אחר מקשים/כפתורים מוחזקים, שחרור אידמפוטנטי

אינם ממציאים policy עסקי מקביל

DisplayPort / Presenter

ציור snapshot וקבלת כוונות משתמש

אינם בעלי State הבטיחות

SessionTelemetry

counters, metrics ודוח מתוך אירועים ותוצאות ביצוע

דוח אינו חלק מתנאי הצלחת cleanup

LiveSession

בניית הרכיבים, חיבורם וניהול lifecycle

אין העברת פונקציית הענק לתוך method ענק

המחלקות התקינות הקיימות, כגון DwellEngine, גלאי המחוות, GazePointFilter ו־ActionRouter, יישמרו וישולבו. אין ליצור InteractionController שמעתיק את החלטות ActionRouter; יש להגדיר בעלות ולהעביר או להאציל אחריות פעם אחת בלבד.

4.3 מסלול הנתונים

מקור המעקב מפרסם FrameObservation. ה־pipeline מחזיר GazeSample ואירועי מחוות מתוך אותו רצף פריימים. ה־SafetyController מפיק הרשאות; ה־InteractionController מציע Actions לפי מצב וכוונה; ה־ActionExecutor בודק מחדש את ההרשאה הרלוונטית לפני ביצוע. התצוגה והטלמטריה צורכות snapshots ותוצאות.

בדיקת תקפות מבט ובדיקת זמינות openness הן נפרדות: פריים שאינו מתאים להזזת סמן עשוי להכיל מידע הדרוש לזיהוי עצימת עיניים. אין לסנן כל מידע המחוות בגלל כשל ב־gaze gate. התנהגות זו תאופיין לפני העברה.

4.4 מבנה קבצים יעד

תיקייה

תכולה

gazelink/domain/

חוזים, enums, יחידות ו־reason codes

gazelink/tracking/

TrackingSource, GazeFollowerSource, ReplaySource

gazelink/gaze/

חיזוי משותף, sample gates, pipeline, preflight

gazelink/calibration/

session, schema, טעינה ושמירה, model contracts

gazelink/interaction/

מחוות, Dwell, תפריט, גלילה, מקלדת, safety/controller/executor

gazelink/platform/

input adapters, screen geometry וגבולות מערכת ההפעלה

gazelink/ui/

display adapters, presentation ו־HUD

gazelink/app/

LiveSession, LiveOptions, configuration וטלמטריה

tools/

הקלטה, אימון, probes, benchmark ו־CLI דקים

tests/

בדיקות ליבה, חוזים, אינטגרציה ו־replay עם תשתיות מדומות

ניתן לחלץ תחילה מודולים בשמות הקיימים ואז להעביר לתיקיות. בסיום התלויות חייבות להתאים למבנה; אין לדחות את ההפרדה בין tools לליבה כעניין קוסמטי. תאימות לפקודות הישנות נשמרת דרך wrappers/re-exports זמניים.

4.5 lifecycle, תהליכונים ושעון

מקור המעקב הוא הבעלים היחיד של המצלמה ושל משאבי הספרייה. רישום cleanup נעשה מיד לאחר כל רכישה מוצלחת, וגם כשל בתוך factory מנקה רכישה חלקית.

LiveSession מקבל source/display factories, input ports ושעון מפורשים. ברירות המחדל נבנות בכניסת האפליקציה; בבדיקות מוזרקים fakes ללא fallback לקלט אמיתי.

callback המצלמה מעדכן pipeline/snapshot מסונכרנים; אין ציור או קלט OS מתוך callback. עדכוני UI נעשים בתהליכון שה־toolkit דורש.

אין תור בלתי מוגבל. מסלול מבט יכול להשתמש ב־latest snapshot; אירועי מחוות מקבלים מזהים ותור מוגבל כדי לא לאבד קריצה בין ticks. overflow/חריגה מתועדים וחוסמים פעולות לא בטוחות. אין לשנות cadence או מדיניות תורים בשקט במהלך החילוץ.

כל השוואת זמן בקרה משתמשת בשעון monotonic מוזרק מאותו תחום זמן. Replay משחזר סדר וזמנים; זמן UTC מיועד לדיווח בלבד.

בסגירה: מסמנים stopping וחוסמים פעולות חדשות באופן מסונכרן; מנסים לשחרר כל קלט מוחזק; עוצרים callback/source וממתינים לסיום מוגבל בזמן; סוגרים תצוגה ומשאבים; רק אז מפיקים דוח. סדר סגירת המשאבים יכבד את תלויות ה־backend.

כל שחרור מוגן בנפרד וממשיך גם אם אחר נכשל. cleanup אידמפוטנטי; שגיאת דוח אינה מונעת שחרור. ExitStack אפשרי, אך סדר LIFO וחריגות נבדקים בפועל. אין הבטחה ש־finally פועל בקריסת תהליך קשיחה או הפסקת חשמל.

5. חוזי נתונים מרכזיים

חוזים יהיו typed dataclasses/Enums, בלתי משתנים ככל האפשר. arrays דורשים בעלות או read-only מפורשים; frozen=True לבדו אינו מגן על buffer משתנה. שדות אופציונליים כוללים reason code. schemas מתועדים וממוספרים; אין reinterpretation של artifacts ישנים.

חוזה

שדות ומשמעות מחייבים

FramePacket

frame_id, capture timestamp, מידות, pixel format ו־buffer בזיכרון בלבד; פנימי ל־Adapter כשאין צורך לחשוף תמונה

FrameObservation

schema/source/frame IDs, captured/observed timestamps עם ציון זמינותם, tracking state, features מספריים מתועדים, openness לכל עין, head pose/geometry אם זמינים, quality ו־reason codes

CalibrationSample

מזהה פריים, זמן, target ומערכת קואורדינטות, features וסכמתם, איכות, accepted/rejected reason; ללא raw frame

GazeSample

frame/time/source IDs, raw ו־filtered normalized XY, screen XY, screen/geometry revision, model/profile revision, quality, valid-for-control ו־reason codes

GestureEvent

event_id, סוג, זמני התחלה/סיום/זיהוי, פריימי מקור ואיכות; טבעי/לא מזוהה אינם הופכים אוטומטית לקליק

Permission

policy revision, זמן הערכה/תוקף, activation ופעולות מותרות, reason codes; נבדק מחדש בביצוע

Action

action_id, סוג, זמן, אירוע מקור, יעד/מקשים/גלילה לפי הצורך, mode/profile revision

ActionResult

בוצע/נדחה/נכשל, reason, זמן ומזהה פעולה; מונע counters של הצלחה עבור פעולה שלא בוצעה

XY מנורמל: origin בפינה שמאלית עליונה, X ימינה ו־Y מטה; 0–1 ביחס למסך המזוהה. מיקום Windows יכול לכלול קואורדינטות שליליות במסכים נוספים. DPI, rotation ו־geometry נפתרים במקום אחד. אין clamp סמוי שמשנה תחזית גולמית; invalid/out-of-screen מדווחים במפורש. confidence שאינו זמין אינו מומצא כהסתברות.

VisionObservation הקיים יישמר או יותאם לחוזה באמצעות migration מפורש; אין צורך להחליף שמות אם הם כבר מייצגים אותו חוזה. features ייחודיים למנוע מתועדים בסכמה ו־model metadata; מבני ספרייה חיצוניים אינם חוצים את הגבול.

הוראת Feature schema 3 של המסלול המקורי נשמרת: iris_in_eye.x לאורך פינות העין; Y הוא היסט ניצב מחולק ברוחב העין ומוזז ב־0.5. עפעפיים משמשים openness וכיוון, ולא עוגני Y. iris_in_lids_y הוא אות משלים/fallback מאומת בלבד. datasets/models/corrections מסכמות 1–2 אינם מתפרשים כסכמה 3. אין לכפות סכמה זו על embeddings של מנוע אחר או להמירם בשקט.

הסעיפים הבאים משמרים את דרישות היכולות מהאפיון המקורי. הם דרישות תפקודיות ולא הצהרה שהמימוש חסר או שהושלם. פרטי מקור/ממשק/בעלות ממומשים לפי סעיפים 3–5 לעיל.

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

נדרש Guided 9-point calibration ששומר CalibrationSample מלא, ולא רק Count. לפי הסקירה קיימים כבר כיול ומודלים: יש לאמת ולשמר אותם, ולהשלים רק פערים בפועל; אין לבנות מחדש מנגנון שכבר עובד.

7.2 שלב B — Quality / Outlier

יש לסנן:

Tracking לא תקין.

Eye visibility לא מספקת.

Head Pose מוחלט חריג (סף קשיח, ייעודי ונדיב יותר לכיול מאשר ל-Live control — ראו M2-02 ב-TASKS.md).

Sample ישן.

Outliers סטטיסטיים בתוך אותה נקודת Target.

כל rejection מקבל Reason code.

הכרעת Outlier נעשית על חלון רציף ולא על Baseline שנבנה מהדגימות הראשונות: Observations שעברו את ה־Gates נשארים provisional עד שנמצא תת־חלון רציף בגודל המינימום שעומד ב־P95 של תנועת שתי העיניים ושל תנועת Head Pose קצרת־טווח. רק החלון שנבחר מתקבל; שאר המועמדים מקבלים target_outlier. אם אין חלון יציב, אותו Target חוזר ל־Stabilization אוטומטית. אין בכך Gate נגד תנוחת ראש שונה לאחר שהמשתמש התייצב על Target אחר.

הערה — Head Pose יחסי ל-Session אינו Gate: הניסוח הקודם כלל גם סינון לפי Head Pose חריג ביחס ל-Session (סטייה יחסית לשאר הכיול, לא סף מוחלט). הוחלט במפורש (M2-02 Stage B) שלא לסנן לפי זה: תנוחת הראש היא Feature שמודל ה-Gaze ב-M2-03 אמור ללמוד ממנה, ולא רעש שיש להסיר, וסינון כזה היה מקטין את מרחב האימון ומגדיל Over-rejection. הסטייה היחסית נחשפת כ-Metric תיאורי בלבד (CalibrationSessionResult.head_pose_spread()), ואינה משפיעה על accepted/rejected.

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

Preflight חמשת הכיוונים של M2 הוא כלי אבחוני לפני Calibration, לא Gate של Control ולא תחליף ל־Validation. הפרדה אופקית דורשת אות מינימלי בכל עין. הפרדה אנכית נבחנת דו־עינית: Combined signal מינימלי, רצפת אות בכל עין והסכמה בסימן; כיוון הפוך או חוסר הסכמה נשארים כשל מפורש. הבדיקה מדווחת אם ההכרעה האנכית הגיעה מ־CORNER_AXIS או מ־LID_RELATIVE; האחרון הוא fallback רק כאשר ציר הפינות לא עבר והוא כן עבר את אותם תנאי הפרדה. הספים הם Placeholders מדודים ומתועדים ב־TASKS.md, נשמרים ב־Artifact ומוצגים בדוח. אין להעתיק את דרישת Head Pose הקבוע של ה־Preflight ל־Calibration, שבו Head Pose הוא Feature חוקי של המודל.

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

OS input כבוי כברירת מחדל. מצב UI בשם ACTIVE אינו הרשאה לקלט OS: activation מפורש ו־Permission עדכני נדרשים בנפרד. יש לשמר את ההפעלה המפורשת הקיימת ולבדוק אותה, ללא שינוי סמוי במיפויי מוצר.

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

מדידות היסטוריות שדווחו ב־2026-09-01 במסלול M1 המקורי; אינן baseline למערכת החיה הנוכחית:

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

15. כללי בטיחות ותאימות בריפקטור

SafetyController הוא מקור האמת להרשאה עסקית. Adapter רשאי לשמור interlock מקומי ויכולת שחרור, אך לא לממש policy עסקי נוסף. אין העברת armed=True קבועה במקום הרשאה אמיתית.

בדיקת freshness מתבצעת גם בזמן הביצוע, לא רק בזמן יצירת האירוע. מזהי אירועים ומצבי session מונעים ביצוע כפול או שימוש באירוע מלפני pause/recovery.

Release ו־Pause הם פעולות הגנה הזמינות גם כשהרשאת קליק/תנועה חסומה; Resume אינו עוקף activation, כיול או תקפות מעקב. אובדן מעקב מבטל צבירת Dwell, מחוות ממתינות וגלילה לפי policy מפורש, ומשחרר קלט מוחזק.

החלפת מסך/geometry או פרופיל/מודל פוסלת snapshots והרשאות ישנים. אין שימוש בנקודות מהגיאומטריה הקודמת. שמור את מנגנוני ההקפאה וה־restart הקיימים.

אין לשנות במסגרת חילוץ: חיתוך/צבע/resize, סדר features, dtype, normalization, מודל ציר X/Y, ספים, preset סינון, סדר הפעלת גלאים או מיפויי מחוות. אין לאמן מחדש מודל כדרך לתקן פער שנוצר בריפקטור.

כפילות live/offline תאוחד רק לאחר אפיון ההבדלים. אם float32/float64 או gates שונים כיום, משתמשים זמנית ב־policy מפורש בתוך ליבה משותפת. איחוד policy שמשנה תוצאה הוא משימת התנהגות נפרדת, ולא שינוי מבנה מוסווה.

גם preflight מחמיר יותר הוא שינוי התנהגות. חילוץ הקוד אינו אישור להחיל אוטומטית את הגרסה המחמירה על כל המסלולים.

artifacts קיימים נשארים קריאים. שינוי schema מחייב migration או כשל ברור עם הסבר. fingerprint יכלול את רכיבי הליבה שחולצו, עם תיעוד השינוי והקשר למודלים הישנים.

default values נמצאים במקור יחיד ומוזרקים דרך LiveOptions/profile. סדר קדימות: CLI מפורש, פרופיל, defaults. ערכים ישנים שונים נשמרים תחילה כ־presets מתועדים; אין ליישרם בשקט.

16. תוכנית ביצוע מחייבת ל־Claude Code

זהו אישור לביצוע הקוד, לא בקשה להחזיר תוכנית נוספת בלבד. העדיפות המיידית היא השלמת הארכיטקטורה. עובדים בשלבים קטנים, ממשיכים אוטונומית כל עוד הבדיקות עוברות, ולא מוסיפים בינתיים יכולות מוצר חדשות ללולאה הישנה.

שלב A — אימות בסיס ותיעוד

קרא את הוראות המאגר, ROADMAP, TASKS ו־ADR הרלוונטיים. אתר את המסלול החי הפעיל ואת מקורות החיזוי בפועל.

השווה את ממצאי הסקירה לקוד. שמור עבודה קיימת של המשתמש; אל תאפס או תדרוס שינויים שלא שייכים למשימה.

עדכן את TECHNICAL_SPEC לפי מסמך זה, הוסף ADR שמחליף את בחירות הארכיטקטורה המיושנות, ועדכן את TASKS ואת קישורי התיעוד הסותרים. שמור היסטוריה במקום למחוק החלטות ישנות.

תעד מצב בדיקות נוכחי ותקלות קודמות. בדיקות הבסיס אינן רשאיות לפלוט קלט OS אמיתי; ודא fakes לפני הרצת תרחישי input. אם חסר גבול בטוח, הוסף את נקודת ההזרקה המינימלית קודם.

שלב B — lifecycle ותיקוני בטיחות מיידיים

תקן cleanup של input לפני דוחות, הגנה עצמאית לכל שחרור, רכישות משאב חלקיות וערבוב שעונים. בדוק כשל באתחול Display אחרי פתיחת המקור, כשל בדוח, כשל ב־release אחד, סגירה כפולה ו־callback שמגיע במהלך shutdown. בכל המקרים אין פעולות חדשות אחרי stopping וכל פעולות הניקוי שנותרו עדיין מנוסות.

תיקון באג בטיחות רשאי לשנות את trace של תרחיש התקלה בלבד; ציין expected-before/after. אין לשמר באג מסוכן כדי לעבור בדיקת זהות. אל תדחה תיקון קטן ודחוף עד הקמת מערך replay מקיף.

שלב C — הזרקת תלויות ובסיס השוואה

הוסף LiveOptions, Clock ו־source/display/input seams. החלף patches של internals ב־fakes מפורשים. אפיין את המסלול הנוכחי על 2–3 הקלטות קיימות, ככל שהן תואמות למודל: gate decisions, prediction, filter output וזמני דגימה. צור trace דטרמיניסטי של פעולות ומעברי מצב באמצעות מקור מדומה ושעון מדומה.

כסה tracking loss/recovery, blink טבעי מול wink, pause/resume, menu, scroll/seed, אירוע ישן או כפול, keyboard target loss, drag ו־Esc בעת קלט מוחזק. אם הקלטות חסרות, השתמש בתרחישים סינתטיים ותעד מה לא ניתן לאמת; אין לדרוש כברירת מחדל סבב מדידות עיניים חדש.

שלב D — חילוץ ליבה מכלי ההקלטה

חלץ Display, חיזוי, model loading, preflight ו־lifecycle מתוך כלי הקלטה/אימון למודולים משותפים. השאר את training orchestration בכלים ואת הדרוש לחיזוי בליבה. שמור wrappers לפקודות ישנות, ועדכן fingerprint ומטא־דאטה כנדרש. הקוד החי לא ייבא את gf_record או gf_fit; הם ייבאו את הליבה.

שלב E — מקור מעקב ו־pipeline משותפים

הכנס חוזי observations ו־GazeFollowerSource. העבר live, recording וכלי המדידה הרלוונטיים לאותו מימוש ליבה, עם policies מפורשים להבדלים הקיימים. השאר סדר נתונים ותזמון שקולים. ReplaySource מריץ ליבה ללא מצלמה. כל שימוש במבני הספרייה החיצונית מחוץ ל־Adapter מוסר.

שלב F — הפרדת אינטראקציה, בטיחות וביצוע

חלץ SafetyController, InteractionController, ActionExecutor, SessionTelemetry ו־Presenter; שלב את המחלקות התקינות הקיימות. העבר את מצב ה־UI לבעלים אחד. הסר את policy הבטיחות הכפול והעבר הרשאה אמיתית ל־executor/adapters. בדוק מצבי איסור, שינוי הרשאה בין החלטה לביצוע, שחרור בזמן pause ו־no-input בזמן stale/low-confidence.

שלב G — חיבור אפליקציה והשלמת מבנה

LiveSession בונה ומנהל את הרכיבים בלבד. run_live וה־CLI הופכים wrappers דקים. רכז configuration, העבר לליבה ול־tools, הסר cycles ויבואי sys.path לא נחוצים. אין לראות במשימה גמורה אם קיימות מחלקות חדשות אך לולאת הענק עדיין מנהלת את כל המצב וההחלטות.

שלב H — אימות וסיום

הרץ את בדיקות הקבלה שלהלן, תקן רגרסיות ועדכן TASKS ו־ADR. דווח מה מומש ומה אומת. בדיקות חומרה שלא ניתן להריץ יירשמו במפורש; הן אינן סיבה לעצור חילוץ עצמאי שניתן לאמת אוטומטית. אין לטעון שהמוצר מוכן לשימוש חי לפני QA חומרה רלוונטי.

17. בדיקות קבלה לארכיטקטורה

תחום

תנאי קבלה

גבולות

אין יבוא של tools מהליבה; אין gazefollower internals מחוץ ל־Adapter; אין pygame/Qt/ctypes במודולי domain ובקרי האינטראקציה

אחריות

בעלים יחיד לכל state/resource בטבלת הרכיבים; לולאת האפליקציה מתזמרת בלבד; אין העתקת פונקציית הענק למחלקה

בדיקות ללא חומרה

LiveSession יכול לרוץ עם ReplaySource/FakeSource, FakeDisplay, FakeInput ו־FakeClock ללא גישה למצלמה או OS input

שקילות מספרית

gate/mask/reason codes זהים; בפעולות שלא השתנה סדרן ובאותה סביבה מצפים לשוויון מספרי מלא; טולרנס אחר רק עם נימוק לפי dtype/פעולות וראיות, לא סף 1e-9 גורף

שקילות התנהגות

אירועים, סדרם, מעברי מצב ותוצאות זהים ב־trace הדטרמיניסטי, למעט תיקוני באגים מתועדים וממוקדים

משאבים

cleanup נבדק בכשל חלקי, כשל release, כשל reporting, עצירה חוזרת ו־callback מאוחר

קלט

אין פעולה ישנה/כפולה/אסורה; releases אינם נחסמים על ידי הרשאת input רגילה; automated tests משתמשים רק ב־FakeInput

תאימות

פקודות קיימות, טעינת פרופילים/מודלים והקלטות קיימות פועלות או מקבלות שגיאת תאימות מפורשת; ללא שינוי שקט בסכמה

תחזוקה

בדיקות imports וגבולות משמעותיות, type/lint ללא רגרסיות בקבצים ששונו; failures קיימים מתועדים ולא מוסווים

ביצועים: יש למדוד את אותו קטע pipeline באותה מכונה, קלט וגרסת תלויות, עם warm-up וכמה חזרות. Replay אינו מוכיח camera-to-cursor latency. עלייה חוזרת של יותר מ־10% ב־p95 משמשת סף חקירה הנדסי לריפקטור, ולא יעד מוצר חדש או אישור להאטה. בדוק גם את השינוי המוחלט במילישניות ואת שונות המדידה לפני הכרעה. תקן overhead מוכח שנוצר בשינוי; כשאין חומרה, סמן את המדד כלא מאומת.

בדיקות ממוקדות בכל חילוץ; suite מלא בסוף שלב ובסיום בהתאם לשערי הפרויקט. חומרה: בדיקה ממוקדת בסיום שינוי המסלול או לפני שימוש חי, simulation תחילה. אין צורך לבקש מהמשתמש כיול חדש לכל שינוי ארכיטקטוני.

