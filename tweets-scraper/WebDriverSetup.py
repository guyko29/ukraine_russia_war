"""
Module for setting up Selenium WebDriver for scraping Twitter (X).

Functions:
- setup_web_driver(): Configures and initializes a Selenium WebDriver instance.

Author: [Your Name]
Date: [Update Date]
"""

import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager as CM
from selenium.webdriver.chrome.options import Options

def setup_web_driver():
    """
    Sets up and initializes a Selenium WebDriver instance for Chrome.
    Authenticates using cookies from cookies.json file.

    Returns:
    - webdriver.Chrome: A Selenium WebDriver instance with Chrome configuration.
    """
    # Configure Chrome options (add custom options here if needed)
    chrome_options = Options()

    # Use WebDriverManager to handle driver installation
    service = Service(executable_path=CM().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # First, navigate to Twitter/X homepage to set the domain
    driver.get('https://x.com')
    time.sleep(2)

    # Load cookies from cookies.json file
    try:
        with open('cookies.json', 'r') as f:
            cookies = json.load(f)

        # Add each cookie to the driver
        for cookie in cookies:
            # Ensure required cookie fields are present
            if 'name' in cookie and 'value' in cookie:
                driver.add_cookie(cookie)
                print(f"Added cookie: {cookie['name']}")

        print("Successfully loaded cookies from cookies.json")
    except FileNotFoundError:
        print("Error: cookies.json file not found!")
    except json.JSONDecodeError:
        print("Error: cookies.json contains invalid JSON!")
    except Exception as e:
        print(f"Error loading cookies: {e}")

    # Navigate to the desired search page (now authenticated)
    driver.get('https://x.com/search?q=%28%23Cake%29+until%3A2019-11-21+since%3A2006-12-17&src=typed_query&f=live')
    time.sleep(5)  # Wait for page to load with authentication

    return driver
