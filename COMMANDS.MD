# GAZELINK — פקודות

כל פקודה נבדקה מול הקוד. עדכון: 16.9.2026.

**סביבות:** הכלים והבדיקות רצים מ-`gazefollower_eval\.venv`. `ruff` ו-`mypy` קיימים רק ב-`focus\.venv`.

```powershell
cd C:\Users\User\Desktop\focus\gazefollower
$py = "C:\Users\User\Desktop\gazefollower_eval\.venv\Scripts\python.exe"
```

---

## התפריט — הפקודה

```powershell
C:\Users\User\Desktop\gazefollower_eval\.venv\Scripts\python.exe gf_live.py --move-cursor --i-mean-it --click-by wink
```

**זו הפקודה היחידה שנותנת תפריט.** התפריט קיים רק כשיש מכונת שליטה, והיא נוצרת רק
ב-`--click-by wink`; זה בתורו מחייב `--move-cursor`, שמחייב `--i-mean-it`. בלי
שלושת הדגלים `gf_live` מציג נקודה בלבד — עצימה ארוכה לא תפתח כלום.

**פותחים בעצימת שתי העיניים לאורך זמן** (עפעוף טבעי הגיע ל-297 מ"ש, עצימה מכוונת
מ-1172 מ"ש; הסף 800 מ"ש), ובוחרים פעולה במבט על אריח. בחירת אריח היא פעולה פנימית —
היא לא שולחת קליק למערכת ההפעלה.

התפריט נפתח גם במצב מושהה, אחרת אין דרך חזרה בלי מקלדת. במצב מושהה שום בחירה בו לא
מגיעה למערכת ההפעלה חוץ מ**המשך שליטה**.

**העין בוחרת את הכפתור** (מ-24.9):

| קריצה | מה נשלח |
|---|---|
| **עין ימין** | קליק **ימני**. תמיד, בלי לבחור כלום |
| **עין שמאל** | קליק **שמאלי**, **יחיד** כברירת מחדל. האריח "לחיצה כפולה" מחליף ללחיצה כפולה וחזרה |

> ⚠️ **הקריצה השמאלית לא נמדדה אצל המשתמש.** היא משתמשת באותם ספים כמו הימנית, בתמונת
> מראה. לפני קליקים אמיתיים בודקים אותה בסימולציה:
> `gf_click_practice.py --click-by wink` (בלי `--click`). בסוף הריצה `winks_by_eye`
> מראה כמה קריצות זוהו מכל עין.

ארבעה אריחים בעמוד, בסדר **ימין לשמאל** (הכיתוב עברית): הראשון למעלה-ימין,
ו-**עוד** תמיד למטה-שמאל ומוביל לעמוד הבא. שלושה עמודים במעגל:

| עמוד | אריח 1 | אריח 2 | אריח 3 | אריח 4 |
|---|---|---|---|---|
| **ראשי** | לחיצה כפולה: כבוי/פעיל | גלילה | מקלדת | עוד ← |
| **ניווט** | חזור | קדימה | החלף חלון | עוד ← |
| **מערכת** | Escape | השהיה / המשך שליטה | סגור תפריט | עוד ← (חזרה לראשי) |

- העמוד "סוג לחיצה" **הוסר**: עין ימין כבר עושה קליק ימני, והמתג מחליף בין יחיד לכפול.
- הדגל `--wink-click single|double` קובע את מצב ההתחלה של הקריצה השמאלית (ברירת מחדל `single`).

- **גלילה** ו**מקלדת** מעבירים למצב, לא מבצעים פעולה בודדת.
- **השהיה/המשך** הוא אותו אריח בשני הכיוונים — כדי שלא תיפול מהמערכת בלי דרך חזרה.
- עצימה ארוכה **פותחת** את התפריט ולא סוגרת אותו; לסגירה יש אריח ייעודי. הסיבה:
  הרמת הסנטר כדי להסתכל על אריח עליון נקראת כשתי עיניים עצומות 800 מ"ש אחר כך.
- מעבר PAUSED/ACTIVE בלי התפריט: **SPACE** (ברירת המחדל).
- `--toggle-by eyes` **מסורב יחד עם התפריט** — אותה מחווה לא יכולה גם לפתוח תפריט וגם
  להחליף מצב. או `--no-menu` ואז העיניים מחליפות מצב, או התפריט ואריח ההשהיה שבו.

---

## המערכת החיה

קלט אמיתי כבוי כברירת מחדל. `--move-cursor` מסורב בלי `--i-mean-it`.

```powershell
# תצוגה בלבד — נקודה על המסך. בלי תפריט, בלי מחוות, בלי מקלדת
& $py C:\Users\User\Desktop\focus\gazefollower\gf_live.py

# בדיקת הסירוב — אמור לסרב, לא לרוץ בסימולציה
& $py C:\Users\User\Desktop\focus\gazefollower\gf_live.py --click-by wink

# עכבר אמיתי, בלי קליקים. עדיין בלי תפריט
& $py C:\Users\User\Desktop\focus\gazefollower\gf_live.py --move-cursor --i-mean-it

# המערכת המלאה: תפריט, מחוות, מקלדת, גלילה וקליקים אמיתיים
# כוון לשולחן עבודה ריק — קליק לא ניתן לביטול
& $py C:\Users\User\Desktop\focus\gazefollower\gf_live.py --move-cursor --i-mean-it --click-by wink
```

`gf_live.py --help` מציג את שאר הדגלים (תזמונים, החלקת עכבר, בחירת מסך).

---

## הקלטה וכיול

```powershell
# מניפסט המתקן: מה היה המצב כשהוקלט (מצלמה)
& $py C:\Users\User\Desktop\focus\gazefollower\gf_setup.py --camera-x-cm 60 --camera-y-cm 63.6 --screen-width-cm 120 --screen-height-cm 33.75

# הקלטת פרוטוקול — החומר שממנו מכיילים (מצלמה)
# --camera-y-cm הוא מיקום המצלמה, והוא מה שמפריד בין מודלים:  -2.0 = מצלמה למעלה (הפעילה),  63.6 = מצלמה למטה.
# מספר הסבב חייב להיות חדש — 47 כבר תפוס.
& $py C:\Users\User\Desktop\focus\gazefollower\gf_record.py --round 48 --protocols A,T1 --camera-x-cm 60 --camera-y-cm -2.0 --screen-width-cm 120 --screen-height-cm 33.75 --overlay-model C:\Users\User\Desktop\focus\gazefollower\recordings\round47\models\group1_ycal_20260924

# התאמת מודל מהקלטות קיימות
& $py C:\Users\User\Desktop\focus\gazefollower\gf_fit.py phase0

# כיול מחדש: מקליט, מתאים, בודק מול הפרופיל הנוכחי ומחליט (מצלמה)
& $py C:\Users\User\Desktop\focus\gazefollower\gf_recalibrate.py

# אותו דבר בלי מצלמה, מהקלטות שכבר על הדיסק
& $py C:\Users\User\Desktop\focus\gazefollower\gf_recalibrate.py --from-existing 46

# תצורות כיול שמורות / מטרות הכיול
& $py C:\Users\User\Desktop\focus\gazefollower\gf_presets.py
& $py C:\Users\User\Desktop\focus\gazefollower\gf_targets.py
```

---

## אימון ו-probes (מצלמה)

```powershell
# מבט, השהיה וקליק אמיתי על כפתורים שרק מקדמים מונה
& $py C:\Users\User\Desktop\focus\gazefollower\gf_click_practice.py                      # בלי קליקים
& $py C:\Users\User\Desktop\focus\gazefollower\gf_click_practice.py --click --i-mean-it  # עם קליקים

# האם הדיוק הזה מאפשר בחירות אמינות
& $py C:\Users\User\Desktop\focus\gazefollower\gf_dwell_practice.py

# איך קריצה נראית על האדם והמצלמה האלה
& $py C:\Users\User\Desktop\focus\gazefollower\gf_wink_probe.py

# האם עצימה מכוונת נפרדת מעפעוף רגיל
& $py C:\Users\User\Desktop\focus\gazefollower\gf_gesture_probe.py

# איזו רזולוציה המצלמה באמת מספקת
& $py C:\Users\User\Desktop\focus\gazefollower\gf_camera_probe.py

# שתי רזולוציות זו לצד זו: איזו נקודה עוקבת טוב יותר
& $py C:\Users\User\Desktop\focus\gazefollower\gf_resolution_view.py
```

---

## ניתוח והשוואה (בלי מצלמה)

```powershell
# מאיפה מגיעה סטייה אופקית
& $py C:\Users\User\Desktop\focus\gazefollower\gf_diagnose_bias.py

# למה הציר האנכי דחוס והאם הכיול מתקן
& $py C:\Users\User\Desktop\focus\gazefollower\gf_vertical.py

# כיול מסנן והערכתו
& $py C:\Users\User\Desktop\focus\gazefollower\gf_filter_benchmark.py

# הקלטה אחת, שני מודלים, אותם פריימים
& $py C:\Users\User\Desktop\focus\gazefollower\gf_recal_compare.py

# האם רזולוציה גבוהה משפרת את המודל
& $py C:\Users\User\Desktop\focus\gazefollower\gf_resolution_compare.py

# האם הפרש במודל מופיע במה שאדם מצליח לבחור
& $py C:\Users\User\Desktop\focus\gazefollower\gf_select_compare.py

# הרצה חוזרת של סשני dwell דרך מנוע הבחירה האמיתי
& $py C:\Users\User\Desktop\focus\gazefollower\gf_selection_replay.py

# מטא-דאטה של המודלים בהשוואת בחירה
& $py C:\Users\User\Desktop\focus\gazefollower\gf_selection_manifest.py

# שיפור המודל המאוגד מהקלטות קיימות
& $py C:\Users\User\Desktop\focus\gazefollower\gf_pool_experiment.py

# ניקוד סשני 24 אינץ' מול הספים
& $py C:\Users\User\Desktop\focus\gazefollower\gf_session.py

# המסכים והרזולוציות שמערכת ההפעלה מדווחת
& $py C:\Users\User\Desktop\focus\gazefollower\gf_screen_check.py
& $py C:\Users\User\Desktop\focus\gazefollower\gf_display.py
```

---

## בדיקות

```powershell
# חובה לפני הרצה (TASKS §1604)
Get-ChildItem -Path C:\Users\User\Desktop\focus\gazefollower -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force

# הסוויטה המלאה — מצופה: Ran 1247 tests / OK
& $py -m unittest discover -s tests

# מודול בודד
& $py -m unittest tests.test_real_input_guard

# golden — מצופה: 3263891C ו-D99D68BD. שינוי = ההתנהגות או התחזיות זזו
Get-FileHash C:\Users\User\Desktop\focus\gazefollower\tests\golden\live_trace.json,C:\Users\User\Desktop\focus\gazefollower\tests\golden\pipeline_digests.json -Algorithm SHA256

# audit קלט אמיתי — מצופה: 10 שורות, כולן מבדיקות ה-guard
$env:GAZELINK_INPUT_REFUSAL_LOG = "C:\Users\User\Desktop\refusals.log"
& $py -m unittest discover -s tests
Get-Content C:\Users\User\Desktop\refusals.log

# lint ו-types — מהסביבה השנייה
& "C:\Users\User\Desktop\focus\.venv\Scripts\ruff.exe" check gazelink_core gf_live.py
& "C:\Users\User\Desktop\focus\.venv\Scripts\python.exe" -m mypy gazelink_core
```

---

## פרטיות

```powershell
# מחיקת קבצים שנגזרו מביומטריה. ברירת המחדל: רשימה בלבד
& $py C:\Users\User\Desktop\focus\gazefollower\gf_purge.py --dry-run
& $py C:\Users\User\Desktop\focus\gazefollower\gf_purge.py --yes
```

---

## סדר בדיקה אחרי שינוי בקוד

1. מחיקת `__pycache__` → סוויטה מלאה (1247, `OK`)
2. golden — שני ה-hash ללא שינוי
3. audit קלט — 10 שורות בלבד
4. ruff + mypy
5. `gf_live` עם מצלמה, בלי דגלים — מעקב, preflight, overlay
6. `gf_live --move-cursor --i-mean-it` — עכבר אמיתי
7. `gf_live --move-cursor --i-mean-it --click-by wink` — **התפריט, המחוות, המקלדת
   והגלילה קיימים רק כאן.** כולל **Esc באמצע קליק מוחזק**

אם מריצים רק אחת מבדיקות החומרה — מספר 7.

**לא נמדד:** זמן השהיה ממצלמה לעכבר, והאם התיקון מפיל קריצות כשזיהוי הפנים מהבהב.
