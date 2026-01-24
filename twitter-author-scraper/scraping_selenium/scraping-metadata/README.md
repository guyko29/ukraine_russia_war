# Twitter Profile Scraper (Selenium) - User Guide

A tool for scraping Twitter/X profile metadata including information from the profile page and the `/about` page.

---

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Getting Twitter Cookies](#getting-twitter-cookies)
4. [Running on Local Machine vs Server](#running-on-local-machine-vs-server)
5. [Usage](#usage)
6. [Input/Output Files](#inputoutput-files)
7. [Smart Rescan Feature](#smart-rescan-feature)
8. [Troubleshooting](#troubleshooting)

---

## Requirements

- Python 3.10 or higher
- Google Chrome browser installed
- ChromeDriver (usually auto-installed with selenium)
- A Twitter/X account

---

## Installation

### 1. Install Python dependencies

Navigate to the project root and install dependencies:

```bash
cd twitter-scraper-author-data
pip install -r requirements.txt
```

### 2. Verify Chrome is installed

```bash
google-chrome --version
# or on some systems:
chromium-browser --version
```

---

## Getting Twitter Cookies

The scraper needs your Twitter login cookies to access profile data. There are two ways to get them:

### Option A: Using Debug Mode (Recommended)

This mode connects to an existing Chrome window where you're already logged in, which bypasses bot detection:

**Step 1:** Start Chrome with debug port:

```bash
# On Linux:
google-chrome --remote-debugging-port=9222 --user-data-dir="$HOME/chrome-selenium-profile"

# On Mac:
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir="$HOME/chrome-selenium-profile"

# On Windows:
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%USERPROFILE%\chrome-selenium-profile"
```

**Step 2:** Login to Twitter in that Chrome window.

**Step 3:** Run the script with `-d` flag:

```bash
cd scraping_selenium/scraping-metadata
python twitter_selenium.py -d
```

This will:
1. Connect to the running Chrome
2. Verify you're logged in
3. Save cookies to `cookies.json`
4. Test with a sample profile

### Option B: Export Cookies from Browser Manually

1. Install the **"EditThisCookie"** browser extension (or similar)
2. Login to Twitter/X in your browser
3. Click the extension icon and find the `auth_token` cookie
4. Create a file named `cookies.json` in the `scraping-metadata` folder:

```json
[
  {
    "name": "auth_token",
    "value": "YOUR_AUTH_TOKEN_VALUE_HERE",
    "domain": ".x.com",
    "path": "/"
  }
]
```

**Important:** The `cookies.json` file must be in the same folder as `run_selenium.py`

---

## Running on Local Machine vs Server

### Local Machine (with Display)

On your local computer, you can:
- Use `-d` mode to get cookies (see above)
- Run the scraper normally (it will use headless mode by default)

### Server (No Display - e.g., Azure, AWS, etc.)

On a server without a display:

1. **Get cookies on your local machine first** using Option A or B above
2. **Copy the `cookies.json` file to the server:**
   ```bash
   scp cookies.json user@server:/path/to/scraping-metadata/
   ```
3. **Run the scraper on the server:**
   ```bash
   python run_selenium.py
   ```

The scraper automatically runs in headless mode, so no display is needed.

---

## Usage

### Basic Usage

```bash
cd scraping_selenium/scraping-metadata
python run_selenium.py
```

### Interactive Prompts

The script will ask you:

1. **Input file:**
   ```
   📂 Use default input file 'all_users.csv'? (y/n): 
   ```
   - Press `y` to use the default file
   - Press `n` to enter a custom file path

2. **Column selection:**
   ```
   Columns:
    a) target_username
    b) other_username
    c) type
   
   Enter column name or letter: b
   ```
   - Enter a letter (a, b, c...) or the full column name

3. **Output file:**
   ```
   💾 Save output to 'all_users_with_data.csv'? (y/n): 
   ```
   - Press `y` to use the default output file
   - Press `n` to enter a custom filename

---

## Input/Output Files

### Input File Format

Your input file (CSV or Excel) should have a column with Twitter usernames:

| other_username |
|----------------|
| elonmusk       |
| @jack          |
| TwitterDev     |

- Usernames can be with or without `@`
- The file can have other columns (they will be ignored)

### Output File Format

The output CSV will have the following columns:

| Column | Source | Description |
|--------|--------|-------------|
| `user_name` | - | The Twitter username |
| `name` | Profile | Display name |
| `bio` | Profile | User biography |
| `location` | Profile | Location from profile |
| `followers` | Profile | Follower count |
| `following` | Profile | Following count |
| `join_date` | About | When the account was created |
| `account_location` | About | Country where account is located |
| `is_verified` | About | True if verified badge |
| `connected_via` | About | App used to create account |
| `username_changes` | About | Number of username changes |
| `uses_vpn` | About | Yes/No based on VPN indicator |

---

## Smart Rescan Feature

The scraper automatically handles resuming and completing partial data:

| Scenario | Action |
|----------|--------|
| User has **all data** (profile + about) | ✓ Skip |
| User has **profile data only** | 📝 Rescan about page only |
| User has **about data only** | 👤 Rescan profile page only |
| User has **no data** | 🔄 Scrape everything |
| User **not in output file** | ➕ Add new entry |

This means you can:
- Stop and resume the scraper at any time
- Run again to fill in missing data
- Add new users to the input file and run again

---

## Troubleshooting

### "Cookies file not found"
- Make sure `cookies.json` is in the `scraping-metadata` folder
- Use `-d` mode on a local machine to create it (see "Getting Twitter Cookies")

### "No data retrieved" for many users
- Your cookies may have expired - get new ones
- Twitter may be rate limiting - the script will automatically take breaks

### "5 consecutive failures" message
- The script will automatically browse Twitter like a human for 3-5 minutes
- This helps avoid detection

### Running on a server
- Make sure Chrome/Chromium is installed:
  ```bash
  sudo apt install chromium-browser  # Ubuntu/Debian
  ```
- No display server (Xvfb) is needed - the script runs headless

---

## Example Session

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

## File Structure

```
scraping-metadata/
├── run_selenium.py      # Main script to run
├── twitter_selenium.py  # Scraper class
├── cookies.json         # Your Twitter cookies (create this!)
├── all_users.csv        # Default input file
└── all_users_with_data.csv  # Default output file
```
