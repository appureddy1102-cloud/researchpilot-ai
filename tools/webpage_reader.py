"""
TOOL B - Page Content Retrieval.

Downloads a web page and extracts the readable text, dropping navigation,
scripts, ads, cookie banners, etc. Unreachable / unsupported pages return
`ok=False` with a reason instead of raising.

Security: the reader refuses non-http(s) URLs and any URL that resolves to a
private/loopback/link-local address (basic SSRF protection), re-checked on
every redirect.
"""
from __future__ import annotations

import ipaddress
import json
import re
import socket
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; ResearchPilotBot/1.0; +https://github.com/)"
MAX_BYTES = 2_000_000
NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form",
              "iframe", "svg", "button", "select", "dialog"]
NOISE_HINT = re.compile(r"(^|[-_ ])(ad|ads|advert|advertisement|banner|cookie|consent|popup|modal|"
                        r"newsletter|subscribe|sidebar|menu|breadcrumb|share|social|related|promo|comment)s?($|[-_ ])", re.I)


def _is_public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def _fail(url: str, error: str) -> dict:
    return {"url": url, "ok": False, "text": "", "title": "", "published_date": None, "error": error}


def read_webpage(url: str, timeout: int = 15, max_chars: int = 9000) -> dict:
    """Fetch and clean one page. Returns dict(url, ok, text, title, published_date, error)."""
    current = url
    try:
        for _ in range(4):  # follow up to 3 redirects manually so each hop is validated
            p = urlparse(current)
            if p.scheme not in ("http", "https") or not p.hostname:
                return _fail(url, "Blocked: only http(s) URLs are allowed.")
            if not _is_public_host(p.hostname):
                return _fail(url, "Blocked: host is not a public internet address.")
            resp = requests.get(current, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.5"},
                                timeout=timeout, stream=True, allow_redirects=False)
            if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                current = urljoin(current, resp.headers["Location"])
                continue
            break
        else:
            return _fail(url, "Too many redirects.")

        if resp.status_code >= 400:
            return _fail(url, f"Page returned HTTP {resp.status_code}.")
        ctype = resp.headers.get("Content-Type", "").lower()
        if "html" not in ctype and "xml" not in ctype and "text/plain" not in ctype:
            return _fail(url, f"Unsupported content type ({ctype.split(';')[0] or 'unknown'}).")
        raw = b""
        for chunk in resp.iter_content(65536):
            raw += chunk
            if len(raw) > MAX_BYTES:
                break
        html = raw.decode(resp.encoding or resp.apparent_encoding or "utf-8", errors="replace")
    except requests.Timeout:
        return _fail(url, f"Timed out after {timeout}s.")
    except requests.RequestException as e:
        return _fail(url, f"Network error: {type(e).__name__}.")

    return extract_content(html, url, max_chars)


def extract_content(html: str, url: str = "", max_chars: int = 9000) -> dict:
    """Pure function (easy to unit-test): HTML -> {title, text, published_date}."""
    soup = BeautifulSoup(html, "html.parser")
    published = _find_published_date(soup)
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()

    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for el in soup.find_all(True):
        if el.attrs is None:
            continue
        hint = " ".join(el.get("class", []) + [el.get("id", "") or ""])
        if hint and NOISE_HINT.search(hint) and el.name not in ("html", "body", "article", "main"):
            el.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup
    parts: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "p", "li"]):
        t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if el.name in ("h1", "h2", "h3"):
            if len(t) > 3:
                parts.append(f"## {t}")
        elif len(t) >= 40:  # skip menu fragments / one-word items
            parts.append(t)
    # de-duplicate repeated lines while keeping order
    seen, lines = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            lines.append(p)
    text = "\n".join(lines)[:max_chars]
    if len(text) < 200:
        return {"url": url, "ok": False, "text": text, "title": title, "published_date": published,
                "error": "Page had too little readable text (possibly paywalled or JavaScript-rendered)."}
    return {"url": url, "ok": True, "text": text, "title": title, "published_date": published, "error": ""}


def _find_published_date(soup: BeautifulSoup) -> str | None:
    for attrs in ({"property": "article:published_time"}, {"name": "article:published_time"},
                  {"property": "og:published_time"}, {"name": "date"}, {"name": "pubdate"},
                  {"itemprop": "datePublished"}, {"name": "citation_publication_date"},
                  {"name": "dc.date"}, {"name": "DC.date.issued"}):
        m = soup.find("meta", attrs=attrs)
        if m and m.get("content"):
            return m["content"].strip()
    t = soup.find("time", attrs={"datetime": True})
    if t:
        return t["datetime"].strip()
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if isinstance(it, dict) and it.get("datePublished"):
                return str(it["datePublished"])
    return None
