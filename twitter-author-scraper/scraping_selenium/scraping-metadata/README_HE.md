# סקרייפר פרופילים טוויטר (Selenium) - מדריך למשתמש

כלי לאיסוף מידע על פרופילים בטוויטר/X, כולל מידע מדף הפרופיל ומדף ה-`/about`.

---

## תוכן עניינים
1. [דרישות מקדימות](#דרישות-מקדימות)
2. [התקנה](#התקנה)
3. [קבלת עוגיות טוויטר](#קבלת-עוגיות-טוויטר)
4. [הרצה במחשב מקומי מול שרת](#הרצה-במחשב-מקומי-מול-שרת)
5. [שימוש](#שימוש)
6. [קבצי קלט/פלט](#קבצי-קלטפלט)
7. [סריקה חכמה](#סריקה-חכמה)
8. [פתרון בעיות](#פתרון-בעיות)

---

## דרישות מקדימות

- Python 3.10 ומעלה
- דפדפן Google Chrome מותקן
- ChromeDriver (בדרך כלל מותקן אוטומטית עם selenium)
- חשבון טוויטר/X

---

## התקנה

### 1. התקנת תלויות Python

נווט לתיקיית הפרויקט והתקן:

```bash
cd twitter-scraper-author-data
pip install -r requirements.txt
```

### 2. ודא ש-Chrome מותקן

```bash
google-chrome --version
# או במערכות מסוימות:
chromium-browser --version
```

---

## קבלת עוגיות טוויטר

הסקרייפר צריך את עוגיות ההתחברות שלך לטוויטר כדי לגשת לנתוני הפרופילים. יש שתי דרכים:

### אפשרות א׳: שימוש במצב Debug (מומלץ)

מצב זה מתחבר לחלון Chrome קיים שבו אתה כבר מחובר, מה שעוקף זיהוי בוטים:

**שלב 1:** פתח Chrome עם debug port:

```bash
# ב-Linux:
google-chrome --remote-debugging-port=9222 --user-data-dir="$HOME/chrome-selenium-profile"

# ב-Mac:
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir="$HOME/chrome-selenium-profile"

# ב-Windows:
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%USERPROFILE%\chrome-selenium-profile"
```

**שלב 2:** התחבר לטוויטר בחלון ה-Chrome הזה.

**שלב 3:** הרץ את הסקריפט עם דגל `-d`:

```bash
cd scraping_selenium/scraping-metadata
python twitter_selenium.py -d
```

זה:
1. יתחבר ל-Chrome הפועל
2. יוודא שאתה מחובר
3. ישמור עוגיות ל-`cookies.json`
4. יבדוק עם פרופיל לדוגמה

### אפשרות ב׳: ייצוא עוגיות מהדפדפן ידנית

1. התקן את התוסף **"EditThisCookie"** לדפדפן (או דומה)
2. התחבר לטוויטר/X בדפדפן שלך
3. לחץ על אייקון התוסף ומצא את העוגייה `auth_token`
4. צור קובץ בשם `cookies.json` בתיקיית `scraping-metadata`:

```json
[
  {
    "name": "auth_token",
    "value": "הערך_של_AUTH_TOKEN_שלך",
    "domain": ".x.com",
    "path": "/"
  }
]
```

**חשוב:** קובץ `cookies.json` חייב להיות באותה תיקייה של `run_selenium.py`

---

## הרצה במחשב מקומי מול שרת

### מחשב מקומי (עם מסך)

במחשב האישי שלך, אפשר:
- להשתמש במצב `-d` כדי לקבל עוגיות (ראה למעלה)
- להריץ את הסקרייפר רגיל (הוא ישתמש במצב headless כברירת מחדל)

### שרת (ללא מסך - למשל Azure, AWS וכו׳)

בשרת ללא מסך:

1. **קבל עוגיות במחשב המקומי שלך קודם** באמצעות אפשרות א׳ או ב׳ למעלה
2. **העתק את קובץ `cookies.json` לשרת:**
   ```bash
   scp cookies.json user@server:/path/to/scraping-metadata/
   ```
3. **הרץ את הסקרייפר בשרת:**
   ```bash
   python run_selenium.py
   ```

הסקרייפר רץ אוטומטית במצב headless, אז אין צורך בשרת תצוגה.

---

## שימוש

### שימוש בסיסי

```bash
cd scraping_selenium/scraping-metadata
python run_selenium.py
```

### שאלות אינטראקטיביות

הסקריפט ישאל אותך:

1. **קובץ קלט:**
   ```
   📂 Use default input file 'all_users.csv'? (y/n): 
   ```
   - לחץ `y` כדי להשתמש בקובץ ברירת המחדל
   - לחץ `n` כדי להזין נתיב לקובץ אחר

2. **בחירת עמודה:**
   ```
   Columns:
    a) target_username
    b) other_username
    c) type
   
   Enter column name or letter: b
   ```
   - הזן אות (a, b, c...) או את שם העמודה המלא

3. **קובץ פלט:**
   ```
   💾 Save output to 'all_users_with_data.csv'? (y/n): 
   ```
   - לחץ `y` כדי להשתמש בקובץ פלט ברירת המחדל
   - לחץ `n` כדי להזין שם קובץ אחר

---

## קבצי קלט/פלט

### פורמט קובץ קלט

קובץ הקלט שלך (CSV או Excel) צריך עמודה עם שמות משתמש טוויטר:

| other_username |
|----------------|
| elonmusk       |
| @jack          |
| TwitterDev     |

- שמות משתמש יכולים להיות עם או בלי `@`
- הקובץ יכול לכלול עמודות נוספות (הן יתעלמו)

### פורמט קובץ פלט

קובץ ה-CSV יכיל את העמודות הבאות:

| עמודה | מקור | תיאור |
|-------|------|-------|
| `user_name` | - | שם המשתמש בטוויטר |
| `name` | פרופיל | שם תצוגה |
| `bio` | פרופיל | ביוגרפיה |
| `location` | פרופיל | מיקום מהפרופיל |
| `followers` | פרופיל | מספר עוקבים |
| `following` | פרופיל | מספר נעקבים |
| `join_date` | אודות | מתי החשבון נוצר |
| `account_location` | אודות | מדינה בה נמצא החשבון |
| `is_verified` | אודות | True אם יש תג אימות |
| `connected_via` | אודות | אפליקציה בה נוצר החשבון |
| `username_changes` | אודות | מספר שינויי שם משתמש |
| `uses_vpn` | אודות | Yes/No לפי אינדיקטור VPN |

---

## סריקה חכמה

הסקרייפר מטפל אוטומטית בהמשכיות ובהשלמת נתונים חלקיים:

| מצב | פעולה |
|-----|-------|
| למשתמש יש **כל הנתונים** (פרופיל + אודות) | ✓ דלג |
| למשתמש יש **רק נתוני פרופיל** | 📝 סרוק רק דף אודות |
| למשתמש יש **רק נתוני אודות** | 👤 סרוק רק דף פרופיל |
| למשתמש **אין נתונים** | 🔄 סרוק הכל |
| משתמש **לא בקובץ הפלט** | ➕ הוסף רשומה חדשה |

זה אומר שאפשר:
- לעצור ולהמשיך את הסקרייפר בכל זמן
- להריץ שוב כדי להשלים נתונים חסרים
- להוסיף משתמשים חדשים לקובץ הקלט ולהריץ שוב

---

## פתרון בעיות

### "Cookies file not found"
- וודא ש-`cookies.json` נמצא בתיקיית `scraping-metadata`
- השתמש במצב `-d` במחשב מקומי כדי ליצור אותו (ראה "קבלת עוגיות טוויטר")

### "No data retrieved" להרבה משתמשים
- העוגיות שלך אולי פגו - קבל חדשות
- טוויטר אולי מגביל בקשות - הסקריפט יקח הפסקות אוטומטית

### הודעת "5 consecutive failures"
- הסקריפט יגלוש אוטומטית בטוויטר כמו משתמש אמיתי במשך 3-5 דקות
- זה עוזר להימנע מזיהוי

### הרצה בשרת
- וודא ש-Chrome/Chromium מותקן:
  ```bash
  sudo apt install chromium-browser  # Ubuntu/Debian
  ```
- אין צורך בשרת תצוגה (Xvfb) - הסקריפט רץ במצב headless

---

## דוגמה להרצה

```
==================================================
🐦 Twitter Profile Scraper (Selenium)
==================================================

📋 Smart Rescan Mode (default):
   ✓ Users with complete data → Skip
   📝 Users missing about data → Rescan about page only
   👤 Users missing profile data → Rescan profile page only
   🔄 Users with no data → Scan everything
   ➕ New users → Scan & append to file

📂 Use default input file 'all_users.csv'? (y/n): y
📂 Using input file: /path/to/all_users.csv

Columns:
 a) target_username
 b) other_username
 c) type

Enter column name or letter: b
Selected column: other_username

💾 Save output to 'all_users_with_data.csv'? (y/n): y
💾 Output file: /path/to/all_users_with_data.csv

📊 Loaded 100 usernames from input file.
🍪 Loading cookies from cookies.json...
   ✅ Added cookie: auth_token to .x.com
✅ Logged in successfully via cookies!

📊 Analysis:
   ✓ Skip (have both): 45
   📝 Rescan about only: 5
   👤 Rescan profile only: 2
   🔄 Rescan all (no data): 8
   ➕ New (not in output): 40

▶️ Starting run for 55 users (out of 100 total).

[1/100] 🔄 Scraping username1 (all)...
👤 Profile username1: {'name': 'User One', 'followers': '1.2K', ...}
📋 About username1: {'join_date': 'March 2020', ...}

...

✅ Done! Results saved to all_users_with_data.csv
```

---

## מבנה קבצים

```
scraping-metadata/
├── run_selenium.py      # הסקריפט הראשי להרצה
├── twitter_selenium.py  # מחלקת הסקרייפר
├── cookies.json         # עוגיות הטוויטר שלך (צור את זה!)
├── all_users.csv        # קובץ קלט ברירת מחדל
└── all_users_with_data.csv  # קובץ פלט ברירת מחדל
```

---

## טיפים

1. **שמור את העוגיות בגיבוי** - הן יכולות לפוג אחרי כמה שבועות
2. **אל תריץ יותר מדי מהר** - הסקרייפר כבר כולל השהיות אקראיות
3. **בדוק את קובץ הפלט תוך כדי** - הנתונים נשמרים כל 10 משתמשים
4. **אם נתקעת** - פשוט הרץ שוב, הסקרייפר ימשיך מאיפה שעצר
