import io
import os
import random
from pathlib import Path

import httpx
import boto3
import bs4
import trio

from otokuna.dumping import add_results_per_page_param, scrape_number_of_pages, add_params
from otokuna.logging import setup_logger


# Initialize logger at module level
logger = setup_logger("dump-svc", include_timestamp=False, propagate=False)

# Realistic User-Agent strings to rotate
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
]


# Sometimes Suumo takes several seconds to respond, but we set a reasonable
# timeout to detect issues early. The function retries 5 times with exponential backoff.
async def get_page(search_url, page, client):
    search_page_url = add_params(search_url, {"page": [str(page)]})
    max_attempts = 5

    for attempt in range(max_attempts):
        try:
            # Add delay before request (except first attempt)
            # Random delay to avoid detection patterns: 1-3 seconds
            if attempt > 0:
                delay = random.uniform(1.0, 3.0)
                logger.info(f"Waiting {delay:.1f}s before retry {attempt + 1}/{max_attempts}")
                await trio.sleep(delay)

            # Rotate User-Agent to look more like real browser traffic
            user_agent = random.choice(USER_AGENTS)

            # Add realistic browser headers
            headers = {
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }

            response = await client.get(
                search_page_url,
                timeout=45.0,  # Increased timeout to 45 seconds
                headers=headers
            )

            # Check HTTP status code
            if response.status_code >= 400:
                logger.error(
                    f"HTTP {response.status_code} for page {page} "
                    f"(attempt {attempt + 1}/{max_attempts}): {search_page_url}"
                )
                # Exponential backoff: 15, 30, 60, 120 seconds
                backoff_delay = min(15 * (2 ** attempt), 120)
                logger.info(f"Backing off for {backoff_delay}s due to HTTP error")
                await trio.sleep(backoff_delay)
                continue

            logger.info(f"Successfully fetched page {page}: {search_page_url}")
            return response

        except httpx.TimeoutException as e:
            logger.error(
                f"Timeout fetching page {page} (attempt {attempt + 1}/{max_attempts}): {e}"
            )
            await trio.sleep(15 * (2 ** attempt))
        except httpx.ConnectError as e:
            logger.error(
                f"Connectivity error fetching page {page} (attempt {attempt + 1}/{max_attempts}): {e}"
            )
            await trio.sleep(15 * (2 ** attempt))
        except httpx.HTTPError as e:
            logger.error(
                f"HTTP error fetching page {page} (attempt {attempt + 1}/{max_attempts}): {e}"
            )
            await trio.sleep(15 * (2 ** attempt))
        except Exception as e:
            logger.error(
                f"Unexpected error fetching page {page} (attempt {attempt + 1}/{max_attempts}): "
                f"{type(e).__name__}: {e}"
            )
            await trio.sleep(15 * (2 ** attempt))

    # All attempts failed
    raise RuntimeError(
        f"Could not fetch page {page} after {max_attempts} attempts. URL: {search_page_url}"
    )


async def get_number_of_pages(search_url, client):
    response = await get_page(search_url, page=1, client=client)
    search_results_soup = bs4.BeautifulSoup(response.text, "html.parser")
    return scrape_number_of_pages(search_results_soup)


async def main_async(event, context):
    output_bucket = os.environ["OUTPUT_BUCKET"]
    batch_name = event.get("batch_name", "")  # (path / '' == path) is True
    base_path = event["base_path"]
    search_url = add_results_per_page_param(event["search_url"])

    dump_path = Path(base_path) / batch_name
    s3_client = boto3.client('s3')
    logger.info(f"Logging properties from batch {batch_name} into: {dump_path}")

    # Create httpx AsyncClient with trio support
    async with httpx.AsyncClient(follow_redirects=True) as http_client:
        max_simultaneous_workers = 5
        limiter = trio.CapacityLimiter(max_simultaneous_workers)
        n_pages = await get_number_of_pages(search_url, http_client)
        logger.info(f"Total result pages: {n_pages}")
        pages = list(range(n_pages, 0, -1))  # pages are 1-indexed

        async def save_page_content(content, bucket, key):
            fileobj = io.BytesIO(content)
            await trio.to_thread.run_sync(s3_client.upload_fileobj, fileobj, bucket, key)

        async def worker(wid):
            while pages:
                page = pages.pop()
                async with limiter:
                    response = await get_page(search_url, page, http_client)
                    logger.info(f"Got page {page} (worker {wid}): {response.url}")
                    key = str(dump_path / f"page_{page:06d}.html")
                    await save_page_content(response.content, output_bucket, key)
                    logger.info(f"Saved to s3 page {page} (worker {wid}): {key}")

        async with trio.open_nursery() as nursery:
            for i in range(max_simultaneous_workers):
                nursery.start_soon(worker, i)

    return event


def main(event, context):
    return trio.run(main_async, event, context)
