from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import json
import os
import sys
import random
import re

# מיפוי חודשים לאנגלית
MONTH_TRANSLATION = {
    'ינואר': 'January', 'פברואר': 'February', 'מרץ': 'March',
    'אפריל': 'April', 'מאי': 'May', 'יוני': 'June',
    'יולי': 'July', 'אוגוסט': 'August', 'ספטמבר': 'September',
    'אוקטובר': 'October', 'נובמבר': 'November', 'דצמבר': 'December',
    'enero': 'January', 'febrero': 'February', 'marzo': 'March',
    'abril': 'April', 'mayo': 'May', 'junio': 'June',
    'julio': 'July', 'agosto': 'August', 'septiembre': 'September',
    'octubre': 'October', 'noviembre': 'November', 'diciembre': 'December',
}


def translate_date_to_english(date_str: str) -> str:
    """Translate date string to English."""
    if not date_str:
        return ""
    result = date_str
    for heb, eng in MONTH_TRANSLATION.items():
        result = result.replace(heb, eng)
    return result


class TwitterSeleniumScraper:
    def __init__(self, cookies_file=None, headless=True, use_debug_port=False):
        """
        Initialize the scraper.
        
        Args:
            cookies_file: Path to cookies.json file
            headless: Run in headless mode (no browser window)
            use_debug_port: Connect to existing Chrome with debug port (bypasses bot detection)
        """
        self.consecutive_failures = 0  # מונה כשלונות רצופים
        self.use_debug_port = use_debug_port
        
        if use_debug_port:
            # Connect to existing Chrome instance with real profile
            # This bypasses bot detection!
            self.options = Options()
            self.options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
            print("🔗 Connecting to existing Chrome on port 9222...")
            self.driver = webdriver.Chrome(options=self.options)
            self.wait = WebDriverWait(self.driver, 15)
            print("✅ Connected to Chrome!")
            return
        
        # Regular mode - create new browser with standard selenium
        self.options = Options()
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Anti-detection measures
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option("useAutomationExtension", False)

        if headless:
            self.options.add_argument("--headless=new")

        self.driver = webdriver.Chrome(options=self.options)
        
        # More anti-detection
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        self.wait = WebDriverWait(self.driver, 15)
        
        if cookies_file and os.path.exists(cookies_file):
            self._load_cookies(cookies_file)

    def _load_cookies(self, cookies_file):
        """Load cookies from JSON file."""
        print(f"🍪 Loading cookies from {cookies_file}...")
        try:
            self.driver.get("https://x.com")
            time.sleep(3)
            
            with open(cookies_file, "r", encoding="utf-8") as f:
                cookies = json.load(f)
            
            for cookie in cookies:
                for domain in ['.x.com', '.twitter.com']:
                    cookie_dict = {
                        'name': cookie['name'],
                        'value': cookie['value'],
                        'domain': domain,
                        'path': cookie.get('path', '/'),
                    }
                    try:
                        self.driver.add_cookie(cookie_dict)
                        print(f"   ✅ Added cookie: {cookie['name']} to {domain}")
                        break
                    except:
                        continue
            
            self.driver.refresh()
            time.sleep(3)
            
            self.driver.get("https://x.com/home")
            time.sleep(3)
            
            if "login" not in self.driver.current_url:
                print("✅ Logged in successfully via cookies!")
                return True
            else:
                print("❌ Cookies didn't work - not logged in")
                return False
                
        except Exception as e:
            print(f"❌ Error loading cookies: {e}")
            return False

    def close(self):
        if self.driver:
            self.driver.quit()

    def save_cookies(self, cookies_file):
        """Save current browser cookies to JSON file."""
        cookies = self.driver.get_cookies()
        with open(cookies_file, "w", encoding="utf-8") as f:
            json.dump(cookies, f, indent=2)
        print(f"✅ Cookies saved to {cookies_file}")

    def manual_login(self, cookies_file="cookies.json"):
        """Open browser for manual login, then save cookies.
        Only works on a machine with display (not server).
        """
        print("🔐 Opening Twitter login page...")
        print("   Please login manually in the browser window.")
        
        self.driver.get("https://x.com/login")
        
        print("\n" + "="*50)
        print("👆 Login to Twitter in the browser window")
        print("   When done, press ENTER here to save cookies")
        print("="*50)
        
        input("\nPress ENTER after you have logged in...")
        
        # בדיקה אם ההתחברות הצליחה
        self.driver.get("https://x.com/home")
        time.sleep(3)
        
        if "login" not in self.driver.current_url:
            self.save_cookies(cookies_file)
            print("✅ Login successful! Cookies saved.")
            return True
        else:
            print("❌ Login failed - still on login page")
            return False

    def random_human_behavior(self):
        """Simulate human-like behavior with random scrolls and pauses."""
        print("   🤖 Simulating human behavior...")
        
        actions = random.choice(['scroll', 'scroll_and_wait', 'visit_random'])
        
        if actions == 'scroll':
            scroll_amount = random.randint(200, 600)
            self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            time.sleep(random.uniform(1, 3))
            self.driver.execute_script(f"window.scrollBy(0, -{scroll_amount // 2});")
            time.sleep(random.uniform(0.5, 1.5))
            
        elif actions == 'scroll_and_wait':
            for _ in range(random.randint(2, 4)):
                scroll = random.randint(100, 400)
                self.driver.execute_script(f"window.scrollBy(0, {scroll});")
                time.sleep(random.uniform(0.5, 2))
            time.sleep(random.uniform(2, 5))
            
        elif actions == 'visit_random':
            self.driver.get("https://x.com/home")
            time.sleep(random.uniform(2, 4))
            for _ in range(random.randint(1, 3)):
                self.driver.execute_script(f"window.scrollBy(0, {random.randint(200, 500)});")
                time.sleep(random.uniform(1, 2))

    def check_and_handle_stuck(self):
        """Check if scraper is stuck and try to unstick it."""
        if self.consecutive_failures >= 2:
            print("   ⚠️ Detected possible block, performing human-like actions...")
            self.random_human_behavior()
            time.sleep(random.uniform(5, 10))
            self.consecutive_failures = 0
            return True
        return False

    def scrape_about_page(self, username: str) -> dict:
        """Scrape the /about page for additional account info."""
        url = f"https://x.com/{username}/about"
        
        about_data = {
            "join_date": "",
            "account_location": "",
            "is_verified": "",  # ריק - לא נכתוב False
            "connected_via": "",
            "username_changes": "",  # עמודה חדשה - מספר שינויי שם
            "uses_vpn": "",  # עמודה חדשה - VPN (מגן) או לא (נעץ)
        }
        
        try:
            time.sleep(random.uniform(1, 3))
            
            self.driver.get(url)
            time.sleep(random.uniform(2, 4))
            
            self.driver.execute_script(f"window.scrollBy(0, {random.randint(50, 150)});")
            time.sleep(random.uniform(0.5, 1))
            
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            lines = [line.strip() for line in page_text.split("\n") if line.strip()]
            
            for i, line in enumerate(lines):
                # תאריך הצטרפות
                if any(m in line for m in ['Joined', 'תאריך הצטרפות', 'Se unió']):
                    if i + 1 < len(lines):
                        date_val = lines[i + 1]
                        about_data["join_date"] = translate_date_to_english(date_val)
                
                # מיקום החשבון
                if any(m in line for m in ['Account located in', 'החשבון נמצא ב', 'Cuenta ubicada en']):
                    if i + 1 < len(lines):
                        about_data["account_location"] = lines[i + 1]
                
                # מחובר דרך
                if any(m in line for m in ['Connected via', 'מחובר דרך', 'Conectado a través de']):
                    if i + 1 < len(lines):
                        about_data["connected_via"] = lines[i + 1]
                
                # מספר שינויי שם משתמש
                if any(m in line for m in ['username change', 'שינוי', 'cambio']):
                    # חפש מספר בשורה
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        about_data["username_changes"] = numbers[0]
            
            # בדיקת VPN - חיפוש ה-SVG בקוד HTML
            try:
                page_html = self.driver.page_source
                
                # path של מגן עם סימן קריאה = VPN
                # מכיל "L10.75 7h2.5L13 12.7" - סימן הקריאה בתוך המגן
                vpn_path = "L10.75 7h2.5L13 12.7"
                
                # path של עיגול עם i = לא VPN (מיקום רגיל)
                # מכיל "M13 17v-5h-2v5h2" - האות i בתוך העיגול
                no_vpn_path = "M13 17v-5h-2v5h2"
                
                if vpn_path in page_html:
                    about_data["uses_vpn"] = "Yes"
                elif no_vpn_path in page_html:
                    about_data["uses_vpn"] = "No"
            except:
                pass
            
            # בדיקת אימות - רק אם יש את הפורמט המדויק "אומת מאז" או "Verified since"
            # זה מופיע רק בחשבונות מאומתים באמת
            for i, line in enumerate(lines):
                # חיפוש הפורמט המדויק: "אומת" בשורה אחת ו"מאז" בשורה הבאה
                if line == 'אומת' or line == 'Verified':
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if 'מאז' in next_line or 'since' in next_line.lower():
                            about_data["is_verified"] = "True"
                            break
            
            print(f"📋 About {username}: {about_data}")
            
        except Exception as e:
            print(f"⚠️ Error scraping about page for {username}: {e}")
        
        return about_data

    def scrape_profile_only(self, username: str) -> dict:
        """Scrape only the profile page (not about page)."""
        profile_data = {
            "name": "",
            "bio": "",
            "location": "",
            "followers": "",
            "following": "",
        }
        
        try:
            time.sleep(random.uniform(1, 3))
            
            self.driver.get(f"https://x.com/{username}")
            time.sleep(random.uniform(2, 4))
            
            self.driver.execute_script(f"window.scrollBy(0, {random.randint(100, 300)});")
            time.sleep(random.uniform(0.5, 1.5))
            
            try:
                name_elem = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='UserName']")
                profile_data["name"] = name_elem.text.split("\n")[0]
            except:
                pass
            
            try:
                bio_elem = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='UserDescription']")
                profile_data["bio"] = bio_elem.text
            except:
                pass
            
            try:
                location_elem = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='UserLocation']")
                profile_data["location"] = location_elem.text
            except:
                pass
            
            # Followers count - try verified_followers first, then regular followers
            try:
                try:
                    followers_link = self.driver.find_element(By.CSS_SELECTOR, "a[href$='/verified_followers']")
                except:
                    followers_link = self.driver.find_element(By.CSS_SELECTOR, "a[href$='/followers']")
                followers_text = followers_link.text
                followers_num = followers_text.split()[0] if followers_text else ""
                profile_data["followers"] = followers_num
            except:
                pass
            
            try:
                following_link = self.driver.find_element(By.CSS_SELECTOR, "a[href$='/following']")
                following_text = following_link.text
                following_num = following_text.split()[0] if following_text else ""
                profile_data["following"] = following_num
            except:
                pass
            
            print(f"👤 Profile {username}: {profile_data}")
            
        except Exception as e:
            print(f"⚠️ Error scraping profile for {username}: {e}")
        
        return profile_data

    def scrape_profile_with_about(self, username: str) -> dict:
        """Scrape profile and about page."""
        profile_data = {
            "user_name": username,
            "name": "",
            "bio": "",
            "location": "",
            "followers": "",
            "following": "",
            "join_date": "",
            "account_location": "",
            "is_verified": "",  # ריק - לא נכתוב False
            "connected_via": "",
            "username_changes": "",
            "uses_vpn": "",
        }
        
        # בדיקה אם נתקענו
        self.check_and_handle_stuck()
        
        got_data = False
        
        try:
            time.sleep(random.uniform(1, 3))
            
            self.driver.get(f"https://x.com/{username}")
            time.sleep(random.uniform(2, 4))
            
            self.driver.execute_script(f"window.scrollBy(0, {random.randint(100, 300)});")
            time.sleep(random.uniform(0.5, 1.5))
            
            try:
                name_elem = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='UserName']")
                profile_data["name"] = name_elem.text.split("\n")[0]
                got_data = True
            except:
                pass
            
            try:
                bio_elem = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='UserDescription']")
                profile_data["bio"] = bio_elem.text
                got_data = True
            except:
                pass
            
            try:
                location_elem = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='UserLocation']")
                profile_data["location"] = location_elem.text
                got_data = True
            except:
                pass
            
            # Followers count - try verified_followers first, then regular followers
            try:
                try:
                    followers_link = self.driver.find_element(By.CSS_SELECTOR, "a[href$='/verified_followers']")
                except:
                    followers_link = self.driver.find_element(By.CSS_SELECTOR, "a[href$='/followers']")
                followers_text = followers_link.text
                # Extract number - format: "1.2K Followers" or "1,234 עוקבים"
                followers_num = followers_text.split()[0] if followers_text else ""
                profile_data["followers"] = followers_num
                got_data = True
            except:
                pass
            
            # Following count
            try:
                following_link = self.driver.find_element(By.CSS_SELECTOR, "a[href$='/following']")
                following_text = following_link.text
                # Extract number - format: "234 Following" or "234 במעקב"
                following_num = following_text.split()[0] if following_text else ""
                profile_data["following"] = following_num
                got_data = True
            except:
                pass
            
            # הדפסת נתוני פרופיל
            profile_info = {k: v for k, v in profile_data.items() if k in ["name", "bio", "location", "followers", "following"] and v}
            if profile_info:
                print(f"👤 Profile {username}: {profile_info}")
            
            # לא בודקים אימות בדף הפרופיל - רק בדף about
            # כי שם יש את הפורמט המדויק "אומת מאז..."
            
            # Get about page data
            about_data = self.scrape_about_page(username)
            
            # עדכון רק ערכים לא ריקים
            for key, val in about_data.items():
                if val:
                    profile_data[key] = val
                    got_data = True
            
            # עדכון מונה כשלונות
            if got_data:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                print(f"   ⚠️ No data retrieved for {username} (failures: {self.consecutive_failures})")
            
        except Exception as e:
            print(f"❌ Error scraping {username}: {e}")
            self.consecutive_failures += 1
        
        return profile_data


# ============ MAIN ============
if __name__ == "__main__":
    debug_mode = "-d" in sys.argv
    cookies_file = "cookies.json"
    
    if debug_mode:
        # מצב debug - מתחבר ל-Chrome קיים עם פרופיל אמיתי
        # עוקף זיהוי רובוטים!
        print("=" * 60)
        print("🔧 DEBUG MODE - Connect to existing Chrome")
        print("=" * 60)
        print()
        print("📋 First, start Chrome with debug port:")
        print()
        print("   On Linux:")
        print('   google-chrome --remote-debugging-port=9222 --user-data-dir="$HOME/chrome-selenium-profile"')
        print()
        print("   On Mac:")
        print('   /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222 --user-data-dir="$HOME/chrome-selenium-profile"')
        print()
        print("   On Windows:")
        print('   "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%USERPROFILE%\\chrome-selenium-profile"')
        print()
        print("📋 Then login to Twitter in that Chrome window.")
        print()
        input("Press ENTER when Chrome is running and you're logged in...")
        
        scraper = TwitterSeleniumScraper(use_debug_port=True)
        
        try:
            # Test that we're logged in
            scraper.driver.get("https://x.com/home")
            time.sleep(3)
            
            if "login" in scraper.driver.current_url:
                print("❌ Not logged in! Please login in Chrome first.")
            else:
                print("✅ Logged in successfully!")
                
                # Save cookies for later use
                scraper.save_cookies(cookies_file)
                
                # Test scrape
                print("\n🧪 Testing with a profile...")
                result = scraper.scrape_profile_with_about("elonmusk")
                print(json.dumps(result, indent=2, ensure_ascii=False))
        finally:
            # Don't close - user's Chrome
            print("\n✅ Done! Chrome window left open.")
    
    else:
        # מצב בדיקה רגיל
        if not os.path.exists(cookies_file):
            print(f"❌ Cookies file not found: {cookies_file}")
            print()
            print("   To create cookies, run with --debug mode:")
            print("   python twitter_selenium.py --debug")
            sys.exit(1)
        
        scraper = TwitterSeleniumScraper(
            cookies_file=cookies_file,
            headless=True
        )
        
        try:
            print("\n🧪 Testing with amichaishilo...")
            result = scraper.scrape_profile_with_about("amichaishilo")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        finally:
            scraper.close()

