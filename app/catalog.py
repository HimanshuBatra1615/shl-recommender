"""
Catalog loader and URL validator.
Every URL returned by the agent MUST exist in the scraped catalog.
This is the hallucination guard.
"""

import json
import logging
from pathlib import Path
from functools import lru_cache

log = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"


@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:
    """Load catalog from JSON. Cached after first call."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found at {CATALOG_PATH}. Run scraper/scraper.py first."
        )
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.info(f"Loaded {len(data)} assessments from catalog")
    return data


@lru_cache(maxsize=1)
def get_url_set() -> frozenset[str]:
    """Return a frozenset of all valid catalog URLs for fast lookup."""
    return frozenset(item["url"] for item in load_catalog())


@lru_cache(maxsize=1)
def get_url_to_item() -> dict[str, dict]:
    """Return a dict mapping URL -> catalog item for fast lookup."""
    return {item["url"]: item for item in load_catalog()}


def validate_url(url: str) -> bool:
    """Return True if URL exists in the scraped catalog."""
    return url in get_url_set()


def get_item_by_url(url: str) -> dict | None:
    """Return catalog item by URL, or None if not found."""
    return get_url_to_item().get(url)


def get_item_by_name(name: str) -> dict | None:
    """Find catalog item by exact or fuzzy name match."""
    name_lower = name.lower().strip()
    for item in load_catalog():
        if item["name"].lower().strip() == name_lower:
            return item
    # Partial match fallback
    for item in load_catalog():
        if name_lower in item["name"].lower():
            return item
    return None


def get_test_type_label(code: str) -> str:
    """Convert single-letter test type code to human-readable label."""
    labels = {
        "A": "Ability & Aptitude",
        "B": "Biodata & Situational Judgement",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
        "K": "Knowledge & Skills",
        "P": "Personality & Behavior",
        "S": "Simulations",
    }
    return labels.get(code, code)
