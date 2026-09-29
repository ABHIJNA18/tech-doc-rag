from pathlib import Path
from collections import Counter
import re


LLMS_PATH = Path("scripts/llms.txt")


CATEGORIES = {
    "support_general": [
        "/support/general-information/",
    ],
    "support_faq": [
        "/support/faq/",
    ],
    "getting_started": [
        "/set-up-ionos-cloud/get-started/",
    ],
    "management": [
        "/set-up-ionos-cloud/management/",
    ],
    "ai": [
        "/ai/",
    ],
    "compute": [
        "/compute-services/",
    ],
    "containers": [
        "/containers/",
    ],
    "storage": [
        "/backup-and-storage/",
    ],
    "networking": [
        "/network-services/",
    ],
    "databases": [
        "/databases/",
    ],
    "data_analytics": [
        "/data-analytics/",
    ],
    "api_reference": [
        "/reference/api-specification-files/",
    ],
}


EXCLUDE_PATTERNS = [
    "/release-notes/",
    "/price-list/",
    "/legal/",
]


SEED_URLS = {
    "https://docs.ionos.com/cloud/support/general-information/service-catalog",
    "https://docs.ionos.com/cloud/support/general-information/contact-information",
    "https://docs.ionos.com/cloud/support/faq/frequently-asked-questions",
}


def extract_urls(text: str) -> list[str]:
    """Extract IONOS Cloud documentation URLs from llms.txt."""

    urls = re.findall(
        r"https://docs\.ionos\.com/cloud/[^\s)]+",
        text,
    )

    # Remove accidental trailing punctuation.
    cleaned = []

    for url in urls:
        url = url.rstrip(".,;:")

        # Normalize URLs ending in .md
        if url.endswith(".md"):
            url = url[:-3]

        cleaned.append(url)

    return sorted(set(cleaned))


def get_category(url: str) -> str | None:
    for category, prefixes in CATEGORIES.items():
        for prefix in prefixes:
            if prefix in url:
                return category

    return None


def is_excluded(url: str) -> bool:
    return any(pattern in url for pattern in EXCLUDE_PATTERNS)


def main():
    if not LLMS_PATH.exists():
        raise FileNotFoundError(
            f"{LLMS_PATH} not found. Run:\n"
            "curl -L https://docs.ionos.com/cloud/llms.txt -o scripts/llms.txt"
        )

    text = LLMS_PATH.read_text(encoding="utf-8")

    urls = extract_urls(text)

    print(f"Total unique URLs: {len(urls):,}")
    print()

    # ---------------------------------------------------------
    # Count URLs by category
    # ---------------------------------------------------------

    counts = Counter()

    for url in urls:
        if is_excluded(url):
            continue

        category = get_category(url)

        if category:
            counts[category] += 1
        else:
            counts["other"] += 1

    print("URL counts by category:")
    print("-" * 40)

    for category, count in counts.most_common():
        print(f"{category:20} {count:5}")

    print()

    # ---------------------------------------------------------
    # Show the three anchor documents
    # ---------------------------------------------------------

    print("Anchor documents:")
    print("-" * 40)

    for url in sorted(SEED_URLS):
        print(url)

    print()

    # ---------------------------------------------------------
    # Show a sample from each useful category
    # ---------------------------------------------------------

    print("Sample URLs:")
    print("-" * 40)

    for category in CATEGORIES:
        matches = [
            url
            for url in urls
            if not is_excluded(url)
            and get_category(url) == category
        ]

        if not matches:
            continue

        print(f"\n[{category}]")

        for url in matches[:5]:
            print(f"  {url}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()