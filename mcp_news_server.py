import xml.etree.ElementTree as ET

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("news-server")

HEADERS = {"User-Agent": "supervisor-multi-agent/1.0 (learning project)"}
WIKI_API = "https://en.wikipedia.org/w/api.php"


@mcp.tool()
def get_latest_news(query: str, max_results: int = 5) -> str:
    """Get the latest news headlines (with source and date) about a topic."""
    r = requests.get(
        "https://news.google.com/rss/search",
        params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
        headers=HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    items = ET.fromstring(r.content).findall("./channel/item")[:max_results]
    if not items:
        return f"No news found for '{query}'."
    return "\n\n".join(
        f"{i.findtext('title')}\nSource: {i.findtext('source', 'unknown')} | "
        f"Date: {i.findtext('pubDate')}\nLink: {i.findtext('link')}"
        for i in items
    )


@mcp.tool()
def wikipedia_search(query: str, max_results: int = 3) -> str:
    """Search Wikipedia and return short summaries for background facts."""
    r = requests.get(
        WIKI_API,
        params={
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": query, "gsrlimit": max_results, "prop": "extracts",
            "exintro": 1, "explaintext": 1, "exsentences": 5,
        },
        headers=HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {}).values()
    pages = sorted(pages, key=lambda p: p.get("index", 0))
    if not pages:
        return f"No Wikipedia results for '{query}'."
    return "\n\n".join(f"{p['title']}\n{p.get('extract', '')}" for p in pages)


@mcp.tool()
def wikipedia_current_events() -> str:
    """Get today's current events from Wikipedia's Current events portal."""
    r = requests.get(
        WIKI_API,
        params={
            "action": "query", "format": "json", "prop": "extracts",
            "explaintext": 1, "titles": "Portal:Current events",
        },
        headers=HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("extract", "No current events found.")[:3000]


if __name__ == "__main__":
    mcp.run()
