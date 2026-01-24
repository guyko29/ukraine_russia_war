import pandas as pd
import os
import time
import sys
import random
from pathlib import Path
from twitter_selenium import TwitterSeleniumScraper

# ---------- CONFIGURATION ----------
script_dir = Path(__file__).parent
DEFAULT_INPUT = "all_users.csv"
DEFAULT_OUTPUT = "all_users_with_data.csv"

BATCH_SIZE = 10
DELAY_BETWEEN_USERS = (2, 5)  # טווח אקראי
DELAY_BETWEEN_BATCHES = (20, 40)  # טווח אקראי
CONSECUTIVE_FAILURES_THRESHOLD = 5  # אחרי כמה כשלונות לקחת הפסקה
LONG_BREAK_MINUTES = (3, 5)  # הפסקה ארוכה בדקות

# עמודות מעודכנות - כולל followers ו-following
FIELDS = [
    "user_name", "name", "bio", "location",
    "followers", "following",
    "join_date", "account_location", "is_verified", "connected_via",
    "username_changes", "uses_vpn"
]

# שדות מדף הפרופיל
PROFILE_FIELDS = ["name", "bio", "location", "followers", "following"]

# שדות מדף האודות
ABOUT_FIELDS = ["join_date", "account_location", "is_verified", "connected_via", "username_changes", "uses_vpn"]
# -----------------------------------


def get_input_file() -> Path:
    """Ask user for input file."""
    default_path = script_dir / DEFAULT_INPUT
    print(f"\n📂 Use default input file '{DEFAULT_INPUT}'? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        if default_path.exists():
            return default_path
        else:
            print(f"❌ Default file not found: {default_path}")
    
    # בקש נתיב
    while True:
        file_path = input("Enter path to input file (.csv or .xlsx): ").strip().strip("'\"")
        if not file_path:
            print("No input given. Exiting.")
            sys.exit(0)
        
        # נסה למצוא את הקובץ
        possible_paths = [
            Path(file_path),
            script_dir / file_path,
            Path.cwd() / file_path
        ]
        for p in possible_paths:
            if p.exists():
                return p
        
        print(f"File not found: {file_path}")


def get_output_file() -> Path:
    """Ask user for output file."""
    default_path = script_dir / DEFAULT_OUTPUT
    print(f"\n💾 Save output to '{DEFAULT_OUTPUT}'? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        return default_path
    
    file_name = input("Enter output filename: ").strip()
    if not file_name:
        return default_path
    
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    
    # שמור בתיקייה הנוכחית של הסקריפט
    return script_dir / file_name


def get_column_name(df: pd.DataFrame) -> str:
    """Ask user for column name."""
    print("\nColumns:")
    for idx, c in enumerate(df.columns):
        letter = chr(ord('a') + idx) if idx < 26 else str(idx)
        print(f" {letter}) {c}")
    
    column_input = input("\nEnter column name or letter: ").strip()
    
    if len(column_input) == 1 and column_input.lower() in 'abcdefghijklmnopqrstuvwxyz':
        col_index = ord(column_input.lower()) - ord('a')
        if col_index < len(df.columns):
            return df.columns[col_index]
    
    return column_input


def load_existing_results(output_file: Path) -> pd.DataFrame:
    """Load existing results from output file."""
    if output_file.exists():
        try:
            return pd.read_csv(output_file)
        except:
            pass
    return pd.DataFrame(columns=FIELDS)


def has_any_data(row: pd.Series) -> bool:
    """Check if a row has ANY meaningful data (not just username).
    Returns True only if at least one field has real data.
    """
    for col in FIELDS:
        if col == "user_name":
            continue
        val = row.get(col, "")
        # בדיקה אם יש ערך אמיתי (לא ריק, לא NaN)
        if pd.notna(val) and str(val).strip() != "":
            return True
    return False


def has_profile_data(row: pd.Series) -> bool:
    """Check if row has data from profile page."""
    for col in PROFILE_FIELDS:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip() != "":
            return True
    return False


def has_about_data(row: pd.Series) -> bool:
    """Check if row has data from about page."""
    for col in ABOUT_FIELDS:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip() != "":
            return True
    return False


def browse_home_like_human(scraper):
    """Browse Twitter home page like a real user."""
    print("   🏠 Browsing Twitter home page like a human...")
    
    try:
        scraper.driver.get("https://x.com/home")
        time.sleep(random.uniform(3, 5))
        
        # גלילות אקראיות כמו משתמש אמיתי
        for _ in range(random.randint(5, 10)):
            action = random.choice(['scroll_down', 'scroll_up', 'pause', 'small_scroll'])
            
            if action == 'scroll_down':
                scroll = random.randint(300, 800)
                scraper.driver.execute_script(f"window.scrollBy(0, {scroll});")
                time.sleep(random.uniform(2, 5))
                
            elif action == 'scroll_up':
                scroll = random.randint(100, 300)
                scraper.driver.execute_script(f"window.scrollBy(0, -{scroll});")
                time.sleep(random.uniform(1, 3))
                
            elif action == 'pause':
                # עצירה לקריאת ציוץ
                time.sleep(random.uniform(3, 8))
                
            elif action == 'small_scroll':
                scroll = random.randint(50, 150)
                scraper.driver.execute_script(f"window.scrollBy(0, {scroll});")
                time.sleep(random.uniform(1, 2))
        
        # חזרה למעלה
        scraper.driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(random.uniform(2, 4))
        
    except Exception as e:
        print(f"   ⚠️ Error browsing home: {e}")


def take_long_break(scraper):
    """Take a long break with human-like browsing."""
    break_minutes = random.uniform(*LONG_BREAK_MINUTES)
    print(f"\n☕ Taking a {break_minutes:.1f} minute break (detected {CONSECUTIVE_FAILURES_THRESHOLD} consecutive failures)...")
    
    # גלישה בדף הבית במהלך חלק מהזמן
    browse_time = random.uniform(30, 90)  # 30-90 שניות של גלישה
    print(f"   📱 Browsing Twitter for {browse_time:.0f} seconds...")
    browse_home_like_human(scraper)
    
    # המתנה את שאר הזמן
    remaining_seconds = (break_minutes * 60) - browse_time
    if remaining_seconds > 0:
        print(f"   💤 Sleeping for {remaining_seconds:.0f} more seconds...")
        time.sleep(remaining_seconds)
    
    print("   ✅ Break finished, resuming scraping...")


def run(usernames: list, output_file: Path, rescan_mode: bool = False):
    """Run the scraper for a list of usernames.
    
    Logic:
    - For each username in input file:
      - If exists in output and has data (more than just username) -> skip
      - If exists in output but only has username (no other data) -> rescan and UPDATE
      - If not in output -> scan and APPEND
    """
    cookies_file = script_dir / "cookies.json"
    
    if not cookies_file.exists():
        print(f"❌ Cookies file not found: {cookies_file}")
        print("   Run 'python twitter_selenium.py --login' on a machine with display to create it.")
        sys.exit(1)
    
    scraper = TwitterSeleniumScraper(
        cookies_file=str(cookies_file),
        headless=True
    )
    
    # Load existing results
    existing_df = load_existing_results(output_file)
    
    # Build lookup: username -> (has_profile, has_about, row_index)
    existing_lookup = {}
    if len(existing_df) > 0:
        for idx, row in existing_df.iterrows():
            username_lower = str(row.get("user_name", "")).lower()
            has_profile = has_profile_data(row)
            has_about = has_about_data(row)
            existing_lookup[username_lower] = {
                "has_profile": has_profile,
                "has_about": has_about,
                "index": idx
            }
    
    # Categorize usernames from input - keep original index
    to_skip = []           # Have both profile and about data
    to_rescan_about = []   # (original_index, username) - Have profile, missing about
    to_rescan_profile = [] # (original_index, username) - Have about, missing profile
    to_rescan_all = []     # (original_index, username) - In output but no data at all
    to_add = []            # (original_index, username) - Not in output
    
    for orig_idx, username in enumerate(usernames):
        username_lower = username.lower()
        if username_lower in existing_lookup:
            info = existing_lookup[username_lower]
            if info["has_profile"] and info["has_about"]:
                to_skip.append(username)
            elif info["has_profile"] and not info["has_about"]:
                to_rescan_about.append((orig_idx, username))
            elif not info["has_profile"] and info["has_about"]:
                to_rescan_profile.append((orig_idx, username))
            else:
                to_rescan_all.append((orig_idx, username))
        else:
            to_add.append((orig_idx, username))
    
    print(f"\n📊 Analysis:")
    print(f"   ✓ Skip (have both): {len(to_skip)}")
    print(f"   📝 Rescan about only: {len(to_rescan_about)}")
    print(f"   👤 Rescan profile only: {len(to_rescan_profile)}")
    print(f"   🔄 Rescan all (no data): {len(to_rescan_all)}")
    print(f"   ➕ New (not in output): {len(to_add)}")
    
    # Combine all users to scrape
    all_to_scrape = []
    for item in to_rescan_about:
        all_to_scrape.append((item[0], item[1], "about_only"))
    for item in to_rescan_profile:
        all_to_scrape.append((item[0], item[1], "profile_only"))
    for item in to_rescan_all:
        all_to_scrape.append((item[0], item[1], "all"))
    for item in to_add:
        all_to_scrape.append((item[0], item[1], "all"))
    
    all_to_scrape.sort(key=lambda x: x[0])  # Sort by original index
    total_users = len(usernames)
    print(f"\n▶️ Starting run for {len(all_to_scrape)} users (out of {total_users} total).")
    
    if len(all_to_scrape) == 0:
        print("✅ All users already have data!")
        scraper.close()
        return
    
    # For updating existing rows, we need to track which ones to update
    results_to_update = []  # (row_index, profile_data, is_partial)
    results_to_append = []  # new profiles to add
    consecutive_failures = 0  # מונה כשלונות רצופים
    
    try:
        for i, (orig_idx, username, scrape_mode) in enumerate(all_to_scrape):
            # בדיקת כשלונות רצופים
            if consecutive_failures >= CONSECUTIVE_FAILURES_THRESHOLD:
                take_long_break(scraper)
                consecutive_failures = 0
            
            mode_icon = {
                "about_only": "📝",
                "profile_only": "👤", 
                "all": "🔄"
            }.get(scrape_mode, "🔄")
            
            print(f"\n[{orig_idx+1}/{total_users}] {mode_icon} Scraping {username} ({scrape_mode})...")
            
            # סריקה לפי המצב
            if scrape_mode == "about_only":
                about_data = scraper.scrape_about_page(username)
                profile = {"user_name": username}
                profile.update(about_data)
            elif scrape_mode == "profile_only":
                profile_data = scraper.scrape_profile_only(username)
                profile = {"user_name": username}
                profile.update(profile_data)
            else:  # all
                profile = scraper.scrape_profile_with_about(username)
            
            # בדיקה אם הצלחנו לקבל נתונים
            got_data = any(
                profile.get(col) and str(profile.get(col)).strip()
                for col in FIELDS if col != "user_name"
            )
            
            if got_data:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                print(f"   ⚠️ No data for {username} (consecutive failures: {consecutive_failures}/{CONSECUTIVE_FAILURES_THRESHOLD})")
            
            username_lower = username.lower()
            if username_lower in existing_lookup:
                # This is a rescan - need to update existing row (partial update)
                row_idx = existing_lookup[username_lower]["index"]
                results_to_update.append((row_idx, profile, scrape_mode != "all"))
            else:
                # New user - append
                results_to_append.append(profile)
            
            if (i + 1) % BATCH_SIZE == 0:
                # Save progress
                save_results_smart(results_to_update, results_to_append, existing_df, output_file)
                results_to_update = []
                results_to_append = []
                # Reload existing_df after save
                existing_df = load_existing_results(output_file)
                
                delay = random.uniform(*DELAY_BETWEEN_BATCHES)
                print(f"💾 Saved batch. Sleeping {delay:.1f}s...")
                time.sleep(delay)
            else:
                time.sleep(random.uniform(*DELAY_BETWEEN_USERS))
        
        # Final save
        if results_to_update or results_to_append:
            save_results_smart(results_to_update, results_to_append, existing_df, output_file)
    
    finally:
        scraper.close()
    
    print(f"\n✅ Done! Results saved to {output_file}")


def save_results_smart(to_update: list, to_append: list, existing_df: pd.DataFrame, output_file: Path):
    """Save results - update existing rows and append new ones.
    For partial updates, only update non-empty values.
    """
    # Update existing rows
    for item in to_update:
        if len(item) == 3:
            row_idx, profile, is_partial = item
        else:
            row_idx, profile = item
            is_partial = False
        
        for col in FIELDS:
            new_val = profile.get(col, "")
            if is_partial:
                # עדכון חלקי - רק אם יש ערך חדש
                if new_val and str(new_val).strip():
                    existing_df.at[row_idx, col] = new_val
            else:
                # עדכון מלא - תמיד מעדכן
                existing_df.at[row_idx, col] = new_val
    
    # Append new rows
    if to_append:
        new_df = pd.DataFrame(to_append, columns=FIELDS)
        existing_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Save entire file
    existing_df.to_csv(output_file, index=False)


def save_results(results: list, output_file: Path):
    """Append results to CSV."""
    df = pd.DataFrame(results, columns=FIELDS)
    
    if output_file.exists():
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, index=False)


def main():
    print("=" * 50)
    print("🐦 Twitter Profile Scraper (Selenium)")
    print("=" * 50)
    
    print("\n📋 Smart Rescan Mode (default):")
    print("   ✓ Users with complete data → Skip")
    print("   📝 Users missing about data → Rescan about page only")
    print("   👤 Users missing profile data → Rescan profile page only")
    print("   🔄 Users with no data → Scan everything")
    print("   ➕ New users → Scan & append to file")
    
    # Get input file
    input_file = get_input_file()
    print(f"📂 Using input file: {input_file}")
    
    # Read input
    if str(input_file).endswith(".csv"):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_excel(input_file)
    
    # Get column name
    column_name = get_column_name(df)
    print(f"Selected column: {column_name}")
    
    # Get output file
    output_file = get_output_file()
    print(f"💾 Output file: {output_file}")
    
    # Extract usernames
    usernames = df[column_name].dropna().astype(str).str.replace("@", "").str.strip().tolist()
    usernames = [u for u in usernames if u]
    
    print(f"\n📊 Loaded {len(usernames)} usernames from input file.")
    
    # Run
    run(usernames, output_file)


if __name__ == "__main__":
    main()
