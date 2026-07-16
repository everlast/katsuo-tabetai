from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import socket
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx
from agents import RunContextWrapper, function_tool
from bs4 import BeautifulSoup

from .context import KatsuoContext
from .models import ScrapedPage

MAX_DOWNLOAD_BYTES = 2_000_000
MAX_STORED_CHARACTERS = 100_000
MAX_TOOL_CHARACTERS = 16_000
USER_AGENT = (
    "Mozilla/5.0 (compatible; katsuo-tabetai/0.1; "
    "+https://github.com/openai/openai-agents-python)"
)


def canonical_url(value: object) -> str:
    parts = urlsplit(str(value))
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()
    port = parts.port
    default_port = (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    netloc = hostname if port is None or default_port else f"{hostname}:{port}"
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


async def _ensure_public_http_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Only public HTTP(S) URLs can be scraped.")
    if parts.username or parts.password:
        raise ValueError("Credential-bearing URLs cannot be scraped.")

    hostname = parts.hostname.casefold()
    if hostname == "localhost" or hostname.endswith(".local"):
        raise ValueError("Local network URLs cannot be scraped.")

    addresses = await asyncio.to_thread(
        socket.getaddrinfo,
        hostname,
        parts.port or (443 if parts.scheme == "https" else 80),
        type=socket.SOCK_STREAM,
    )
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Private or reserved network URLs cannot be scraped.")


def _extract_readable_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if soup.head is not None:
        soup.head.decompose()
    for element in soup(["script", "style", "noscript", "template", "svg"]):
        element.decompose()

    lines: list[str] = []
    previous = ""
    for raw_line in soup.get_text("\n").splitlines():
        line = " ".join(raw_line.split())
        if not line or line == previous:
            continue
        lines.append(line)
        previous = line
    content = "\n".join(lines).strip()
    return title[:500], content[:MAX_STORED_CHARACTERS]


async def fetch_scraped_page(url: str) -> ScrapedPage:
    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.8"},
    ) as client:
        current_url = url
        for _ in range(6):
            await _ensure_public_http_url(current_url)
            async with client.stream("GET", current_url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Redirect response did not include a location.")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                chunks: list[bytes] = []
                byte_count = 0
                async for chunk in response.aiter_bytes():
                    byte_count += len(chunk)
                    if byte_count > MAX_DOWNLOAD_BYTES:
                        raise ValueError("Scraped page exceeds the download size limit.")
                    chunks.append(chunk)
                body = b"".join(chunks)
                content_type = response.headers.get("content-type", "").casefold()
                if "html" not in content_type:
                    raise ValueError(
                        f"Unsupported scraped content type: {content_type or 'unknown'}"
                    )
                encoding = response.encoding or "utf-8"
                html = body.decode(encoding, errors="replace")
                title, content = _extract_readable_text(html)
                if len(content) < 40:
                    raise ValueError("Scraped page did not contain enough readable text.")
                return ScrapedPage(
                    requested_url=url,
                    final_url=str(response.url),
                    fetched_at=datetime.now(timezone.utc),
                    status_code=response.status_code,
                    title=title,
                    content=content,
                    content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                )
        raise ValueError("Scraped page exceeded the redirect limit.")


@function_tool
async def scrape_reference_page(
    wrapper: RunContextWrapper[KatsuoContext],
    url: str,
) -> str:
    """Fetch a public restaurant or review page and return readable source text."""
    try:
        page = await fetch_scraped_page(url)
    except (httpx.HTTPError, OSError, ValueError) as exc:
        return json.dumps(
            {"status": "error", "url": url, "error": str(exc)},
            ensure_ascii=False,
        )

    wrapper.context.scraped_pages[canonical_url(page.requested_url)] = page
    wrapper.context.scrape_calls += 1
    payload = page.model_dump(mode="json", exclude={"content"})
    payload["status"] = "fetched"
    payload["content"] = page.content[:MAX_TOOL_CHARACTERS]
    payload["content_truncated"] = len(page.content) > MAX_TOOL_CHARACTERS
    return json.dumps(payload, ensure_ascii=False)
