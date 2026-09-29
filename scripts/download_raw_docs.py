import asyncio
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx


LLMS_PATH = Path("scripts/llms.txt")

RAW_DIR = Path.home() / "Projects" / "ionos-docs" / "raw"

CONCURRENCY = 5
REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 3


# ---------------------------------------------------------
# Categories we want
# ---------------------------------------------------------

CATEGORIES = {

    # CUSTOMER SUPPORT — download EVERYTHING

    "support_general": "/support/general-information/",

    "support_faq": "/support/faq/",

    # Related technical context

    "getting_started": "/set-up-ionos-cloud/get-started/",

    "management": "/set-up-ionos-cloud/management/",

    "ai": "/ai/",

    "compute": "/compute-services/",

    "containers": "/containers/",

    "storage": "/backup-and-storage/",

    "networking": "/network-services/",

    "databases": "/databases/",

    "data_analytics": "/data-analytics/",

    "api_reference": "/reference/api-specification-files/",

}

# Maximum pages from each category.
# This keeps the first corpus manageable.
CATEGORY_LIMITS = {
    "getting_started": 10,
    "management": 10,
    "ai": 15,
    "compute": 15,
    "containers": 10,
    "storage": 15,
    "networking": 10,
    "databases": 10,
    "data_analytics": 5,
    "api_reference": 10,
}


SEED_URLS = {
    "https://docs.ionos.com/cloud/support/general-information/service-catalog",
    "https://docs.ionos.com/cloud/support/general-information/contact-information",
    "https://docs.ionos.com/cloud/support/faq/frequently-asked-questions",
}


EXCLUDE_PATTERNS = [
    "/release-notes/",
    "/price-list/",
    "/legal/",
]


def extract_urls(text: str) -> list[str]:
    urls = re.findall(
        r"https://docs\.ionos\.com/cloud/[^\s)]+",
        text,
    )

    cleaned = []

    for url in urls:
        url = url.rstrip(".,;:")

        if url.endswith(".md"):
            url = url[:-3]

        cleaned.append(url)

    return sorted(set(cleaned))


def is_excluded(url: str) -> bool:
    return any(pattern in url for pattern in EXCLUDE_PATTERNS)


def get_category(url: str) -> str | None:
    for category, prefix in CATEGORIES.items():
        if prefix in url:
            return category

    return None


def select_urls(urls: list[str]) -> list[str]:
    selected = set()

    for category, prefix in CATEGORIES.items():

        matches = [
            url
            for url in urls
            if prefix in url
            and not is_excluded(url)
        ]

        matches = sorted(matches)

        # -------------------------------------------------
        # IMPORTANT:
        # Download ALL Support + FAQ pages.
        # -------------------------------------------------

        if category in {
            "support_general",
            "support_faq",
        }:
            selected.update(matches)

        # -------------------------------------------------
        # Technical sections remain bounded for now.
        # -------------------------------------------------

        else:
            limit = CATEGORY_LIMITS[category]
            selected.update(matches[:limit])

    # Always include our explicitly chosen anchor pages.
    selected.update(SEED_URLS)

    return sorted(selected)

def raw_path(url: str) -> Path:
    """
    Convert:

    https://docs.ionos.com/cloud/compute-services/...

    into:

    ~/Projects/ionos-docs/raw/compute-services/....md
    """

    parsed = urlparse(url)

    path = parsed.path

    if path.startswith("/cloud/"):
        path = path[len("/cloud/"):]

    if not path.endswith(".md"):
        path += ".md"

    return RAW_DIR / path


async def download_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
):
    async with semaphore:

        markdown_url = url + ".md"

        output_path = raw_path(url)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Don't download again if already present.
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"SKIP  {url}")
            return True

        for attempt in range(1, MAX_RETRIES + 1):

            try:
                response = await client.get(markdown_url)

                response.raise_for_status()

                content = response.text

                if not content.strip():
                    raise ValueError("Empty response")

                output_path.write_text(
                    content,
                    encoding="utf-8",
                )

                print(f"OK    {url}")

                return True

            except Exception as exc:

                print(
                    f"FAIL  {url} "
                    f"(attempt {attempt}/{MAX_RETRIES}): {exc}"
                )

                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)

        return False


async def main():
    if not LLMS_PATH.exists():
        raise FileNotFoundError(
            f"{LLMS_PATH} not found.\n"
            "Run:\n"
            "curl -L https://docs.ionos.com/cloud/llms.txt "
            "-o scripts/llms.txt"
        )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = LLMS_PATH.read_text(
        encoding="utf-8"
    )

    urls = extract_urls(text)

    selected = select_urls(urls)

    print()
    print(f"Total URLs in index: {len(urls):,}")
    print(f"Selected URLs:       {len(selected):,}")
    print(f"Raw directory:       {RAW_DIR}")
    print()

    # Print category breakdown
    category_counts = {}

    for url in selected:
        category = get_category(url) or "seed/other"
        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    print("Selected corpus:")
    print("-" * 40)

    for category, count in sorted(category_counts.items()):
        print(f"{category:20} {count:5}")

    print()

    semaphore = asyncio.Semaphore(CONCURRENCY)

    headers = {
        "User-Agent": (
            "tech-doc-rag/0.1 "
            "(research/learning project)"
        )
    }

    timeout = httpx.Timeout(
        REQUEST_TIMEOUT
    )

    async with httpx.AsyncClient(
        headers=headers,
        timeout=timeout,
        follow_redirects=True,
    ) as client:

        tasks = [
            download_one(
                client,
                semaphore,
                url,
            )
            for url in selected
        ]

        results = await asyncio.gather(
            *tasks
        )

    successful = sum(results)
    failed = len(results) - successful

    print()
    print("=" * 50)
    print("Download complete")
    print("=" * 50)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Raw docs:   {RAW_DIR}")


if __name__ == "__main__":
    asyncio.run(main())