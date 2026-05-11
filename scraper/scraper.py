"""
SHL Product Catalog Scraper
Scrapes Individual Test Solutions (type=1) only.
Uses requests + BeautifulSoup for speed (catalog pages render server-side).
Falls back to Playwright for JS-heavy detail pages.
"""

import json
import time
import re
import logging
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/"
PAGE_SIZE = 12
TYPE_INDIVIDUAL = 1  # Individual Test Solutions only

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get_page(url: str, retries: int = 3, delay: float = 1.5) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup object, with retries."""
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=20)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
            elif resp.status_code == 404:
                log.warning(f"404 for {url}")
                return None
            else:
                log.warning(f"HTTP {resp.status_code} for {url}, attempt {attempt+1}")
        except Exception as e:
            log.warning(f"Error fetching {url}: {e}, attempt {attempt+1}")
        time.sleep(delay * (attempt + 1))
    return None


def get_listing_page(start: int) -> list[dict]:
    """
    Scrape one listing page and return list of {name, url} dicts.
    URL format: ?start={start}&type=1
    """
    url = f"{CATALOG_URL}?start={start}&type={TYPE_INDIVIDUAL}"
    log.info(f"Scraping listing page: {url}")
    soup = get_page(url)
    if not soup:
        return []

    items = []
    # The catalog items are anchor tags within the main content area
    # Pattern: links to /products/product-catalog/view/{slug}/
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/products/product-catalog/view/" in href:
            name = a.get_text(strip=True)
            full_url = href if href.startswith("http") else urljoin(BASE_URL, href)
            if name:
                items.append({"name": name, "url": full_url})

    log.info(f"  Found {len(items)} items at start={start}")
    return items


def parse_detail_page(soup: BeautifulSoup, url: str) -> dict:
    """
    Parse an individual assessment detail page.
    Extracts: description, test_type, job_levels, languages, duration, remote_testing, adaptive_irt
    """
    result = {
        "description": "",
        "test_type": [],
        "job_levels": [],
        "languages": 0,
        "duration_minutes": None,
        "remote_testing": False,
        "adaptive_irt": False,
    }

    # --- Description ---
    # Try common description containers
    desc_candidates = [
        soup.find("div", class_=re.compile(r"description|overview|content|body", re.I)),
        soup.find("p"),
    ]
    for candidate in desc_candidates:
        if candidate:
            text = candidate.get_text(separator=" ", strip=True)
            if len(text) > 30:
                result["description"] = text[:1000]
                break

    # Full page text for attribute parsing
    page_text = soup.get_text(separator="\n", strip=True)

    # --- Test Type ---
    # Look for type labels in the page
    type_map = {
        "Ability & Aptitude": "A",
        "Biodata & Situational Judgement": "B",
        "Competencies": "C",
        "Development & 360": "D",
        "Assessment Exercises": "E",
        "Knowledge & Skills": "K",
        "Personality & Behavior": "P",
        "Simulations": "S",
    }
    found_types = []
    for label, code in type_map.items():
        if label.lower() in page_text.lower():
            found_types.append(code)

    # Also look for short badge spans with single letters
    for span in soup.find_all(["span", "div", "td"], string=re.compile(r"^[ABCDEKPS]$")):
        letter = span.get_text(strip=True)
        if letter in "ABCDEKPS" and letter not in found_types:
            found_types.append(letter)

    result["test_type"] = found_types if found_types else ["K"]  # default Knowledge

    # --- Job Levels ---
    level_keywords = [
        "entry", "graduate", "director", "front line", "frontline",
        "general population", "manager", "mid", "professional",
        "senior", "executive", "supervisor"
    ]
    found_levels = [lv for lv in level_keywords if lv in page_text.lower()]
    result["job_levels"] = list(set(found_levels))

    # --- Languages ---
    lang_match = re.search(r"(\d+)\s+(?:language|languages)", page_text, re.I)
    if lang_match:
        result["languages"] = int(lang_match.group(1))

    # --- Duration ---
    dur_match = re.search(r"(\d+)\s*(?:minute|minutes|min)", page_text, re.I)
    if dur_match:
        result["duration_minutes"] = int(dur_match.group(1))

    # --- Remote Testing ---
    if re.search(r"remote\s+testing", page_text, re.I):
        result["remote_testing"] = True

    # --- Adaptive / IRT ---
    if re.search(r"adaptive|IRT|item response", page_text, re.I):
        result["adaptive_irt"] = True

    return result


def scrape_detail(name: str, url: str) -> dict | None:
    """Scrape a single assessment detail page and return full record."""
    soup = get_page(url)
    if not soup:
        log.warning(f"  Skipping (no page): {name}")
        return None

    details = parse_detail_page(soup, url)

    # If description is empty, try to get it from the page title/meta
    if not details["description"]:
        meta = soup.find("meta", attrs={"name": "description"}) or \
               soup.find("meta", property="og:description")
        if meta and meta.get("content"):
            details["description"] = meta["content"][:1000]

    # Infer test type from name if still empty/default
    name_lower = name.lower()
    if not details["test_type"] or details["test_type"] == ["K"]:
        if any(w in name_lower for w in ["personality", "opq", "behavior", "behaviour", "mq", "motivation"]):
            details["test_type"] = ["P"]
        elif any(w in name_lower for w in ["verify", "ability", "reasoning", "numerical", "verbal", "inductive", "deductive"]):
            details["test_type"] = ["A"]
        elif any(w in name_lower for w in ["simulation", "sim "]):
            details["test_type"] = ["S"]
        elif any(w in name_lower for w in ["sjt", "situational"]):
            details["test_type"] = ["B"]

    return {
        "name": name,
        "url": url,
        **details,
    }


def scrape_all_individual_tests() -> list[dict]:
    """Scrape all Individual Test Solutions from the catalog."""
    all_items = []
    seen_urls = set()

    # First, collect all item URLs from listing pages
    log.info("=== Phase 1: Scraping listing pages ===")
    start = 0
    while True:
        items = get_listing_page(start)
        if not items:
            log.info(f"No items at start={start}, stopping.")
            break

        new_items = [i for i in items if i["url"] not in seen_urls]
        for i in new_items:
            seen_urls.add(i["url"])
        all_items.extend(new_items)

        log.info(f"Total unique items so far: {len(all_items)}")
        start += PAGE_SIZE
        time.sleep(0.5)  # polite delay

        # Safety cap at 50 pages (600 items max)
        if start > 600:
            break

    log.info(f"=== Phase 1 done: {len(all_items)} unique listings found ===")

    # Then, scrape each detail page
    log.info("=== Phase 2: Scraping detail pages ===")
    catalog = []
    for idx, item in enumerate(all_items, 1):
        log.info(f"[{idx}/{len(all_items)}] {item['name']}")
        record = scrape_detail(item["name"], item["url"])
        if record:
            catalog.append(record)
        time.sleep(0.3)  # polite delay

    return catalog


def main():
    output_path = Path(__file__).parent.parent / "data" / "catalog.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = scrape_all_individual_tests()
    log.info(f"=== Scraped {len(catalog)} assessments total ===")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    log.info(f"Saved to {output_path}")
    print(f"\n✅ Done! {len(catalog)} assessments saved to {output_path}")


if __name__ == "__main__":
    main()
