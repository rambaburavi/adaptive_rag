import json
import time
from pathlib import Path

import wikipediaapi


# --------------------------------------------------
# Configuration
# --------------------------------------------------

OUTPUT_DIR = Path("data/corpus")
TARGET_ARTICLES = 250

CATEGORIES = [
    "Artificial intelligence",
    "Machine learning",
    "Computer science",
]


# --------------------------------------------------
# Wikipedia setup
# --------------------------------------------------

wiki = wikipediaapi.Wikipedia(
    user_agent="AdaptiveRAGController/1.0 (educational project)",
    language="en",
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def clean_text(text):
    """Clean unnecessary whitespace."""
    return " ".join(text.split())


def save_article(article):
    """Save one Wikipedia article as JSON."""

    filename = OUTPUT_DIR / f"{article.pageid}.json"

    data = {
        "title": article.title,
        "pageid": article.pageid,
        "url": article.fullurl,
        "text": clean_text(article.text),
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collect_category(category_name, articles):
    """
    Collect pages from a Wikipedia category.
    """

    print(f"\nCollecting category: {category_name}")

    category = wiki.page(f"Category:{category_name}")

    if not category.exists():
        print(f"Category not found: {category_name}")
        return

    for page in category.categorymembers.values():

        if len(articles) >= TARGET_ARTICLES:
            break

        # Only collect actual Wikipedia articles
        if page.ns != wikipediaapi.Namespace.MAIN:
            continue

        if page.pageid in articles:
            continue

        if not page.exists():
            continue

        text = page.text.strip()

        # Ignore extremely small pages
        if len(text) < 1000:
            continue

        articles[page.pageid] = page

        print(
            f"[{len(articles)}/{TARGET_ARTICLES}] "
            f"{page.title}"
        )

        save_article(page)

        time.sleep(0.1)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    articles = {}

    for category in CATEGORIES:

        if len(articles) >= TARGET_ARTICLES:
            break

        collect_category(category, articles)

    print("\n" + "=" * 60)
    print("Wikipedia collection complete")
    print("=" * 60)

    print(f"Total articles collected: {len(articles)}")
    print(f"Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()