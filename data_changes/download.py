import os
import csv
import requests
from urllib.parse import urlparse

# ----------------- CONFIG -----------------
csv_file = "photo_links.csv"
downloaded_log = "downloaded.csv"
base_dir = "photos"
os.makedirs(base_dir, exist_ok=True)

# ----------------- LOAD DOWNLOAD HISTORY -----------------
def load_downloaded(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline='') as f:
        return set(row['img_url'] for row in csv.DictReader(f))

def save_downloaded(path, downloaded):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["img_url"])
        writer.writeheader()
        for url in downloaded:
            writer.writerow({"img_url": url})

# ----------------- DOWNLOAD -----------------
downloaded_urls = load_downloaded(downloaded_log)
total_downloaded = 0

# Counter per beer_slug
beer_counters = {}

with open(csv_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        beer_type = row['beer_type']
        beer_slug = row['beer_slug']
        img_url = row['img_url']

        if img_url in downloaded_urls:
            print(f"🚫 Skipping (already downloaded): {img_url}")
            continue

        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            beer_path = os.path.join(base_dir, beer_type, beer_slug)
            os.makedirs(beer_path, exist_ok=True)

            if beer_slug not in beer_counters:
                beer_counters[beer_slug] = 1
            index = beer_counters[beer_slug]
            filename = f"{beer_slug}_{index:03d}.jpg"
            beer_counters[beer_slug] += 1

            save_path = os.path.join(beer_path, filename)

            with open(save_path, 'wb') as out_file:
                out_file.write(response.content)

            print(f"✓ Downloaded: {save_path}")
            downloaded_urls.add(img_url)
            total_downloaded += 1

        except Exception as e:
            print(f"✗ Failed: {img_url} — {e}")

# ----------------- SAVE LOG -----------------
save_downloaded(downloaded_log, downloaded_urls)
print(f"\n✅ Done. Total downloaded: {total_downloaded}")
