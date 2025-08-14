import os
import time
import csv
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ----------------- CONFIG -----------------
beer_types = ["lager", "wheat","stout","ipa","cider"]
max_photos_per_beer = 300
max_photos_per_type = 1000
output_csv = "photo_links.csv"
os.makedirs("photos", exist_ok=True)

# ----------------- SETUP ------------------
options = Options()
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

# ----------------- MANUAL LOGIN ------------------
def manual_login():
    driver.get("https://untappd.com/login")
    print("🔐 Please complete login manually (including CAPTCHA).")
    input("✅ Press Enter here AFTER you've logged in and see your profile on Untappd...")
    try:
        user_link = driver.find_element(By.CSS_SELECTOR, "a[href^='/user/']")
        print(f"✅ Logged in as: {user_link.get_attribute('href')}")
    except:
        driver.save_screenshot("login_fail.png")
        print("❌ Login failed. Screenshot saved as login_fail.png")
        driver.quit()
        exit(1)

# ----------------- SCRAPE ONE TYPE ------------------
def collect_photo_links_for_type(beer_type, writer):
    driver.get(f"https://untappd.com/search?q={beer_type}&type=beer")

    try:
        sort_dropdown = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "sort-selector"))) #Search by type
        sort_dropdown.click()
        most_popular = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[text()='Most Popular']"))) #Sort by most popular
        most_popular.click()
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ Sorting failed for {beer_type}: {e}")

    beer_links = []
    cards = driver.find_elements(By.CSS_SELECTOR, "a[href^='/b/']")
    for card in cards:
        href = card.get_attribute("href")
        if href and href.count("/") == 5 and "/photos" not in href:
            beer_links.append(href)
    beer_links = list(dict.fromkeys(beer_links))

    print(f"🔗 [{beer_type}] Found {len(beer_links)} beer links")

    type_collected = 0
    seen_global_ids = set()

    for beer_url in beer_links:
        if type_collected >= max_photos_per_type:
            break

        beer_slug = beer_url.split("/b/")[-1].split("/")[0]
        print(f"\n➡️ [{beer_type}] Visiting: {beer_url}/photos")
        try:
            driver.get(beer_url + "/photos")
        except Exception as e:
            print(f"❌ Failed to open {beer_url}/photos: {e}")
            continue

        seen_ids = set()
        image_urls = []

        while len(image_urls) < max_photos_per_beer and type_collected < max_photos_per_type: #Gather links
            soup = BeautifulSoup(driver.page_source, "html.parser")
            new_blocks = 0
            for div in soup.find_all("div", id=lambda x: x and x.startswith("photoJSON_")):
                if div["id"] not in seen_ids and div["id"] not in seen_global_ids:
                    seen_ids.add(div["id"])
                    seen_global_ids.add(div["id"])
                    try:
                        data = json.loads(div.text.strip())
                        img_url = data["photo"]["photo_img_og"]
                        image_urls.append(img_url)
                        new_blocks += 1
                    except:
                        pass

            print(f"📸 [{beer_type}] {beer_slug}: {len(image_urls)} / {max_photos_per_beer} (type total: {type_collected})")

            if type_collected >= max_photos_per_type or new_blocks == 0:
                break

            try:
                show_more = driver.find_element(By.CSS_SELECTOR, "a.more_beer_photos")
                driver.execute_script("arguments[0].click();", show_more)
                time.sleep(2)
            except:
                break

        for url in image_urls:
            writer.writerow([beer_type, beer_slug, url])
            type_collected += 1
            if type_collected >= max_photos_per_type:
                break

    print(f"✅ Done collecting for {beer_type}: {type_collected} links")

# ----------------- RUN ------------------
manual_login()
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["beer_type", "beer_slug", "img_url"])
    for beer_type in beer_types:
        collect_photo_links_for_type(beer_type, writer)

driver.quit()
print("\n✅ All photo links collected.")
