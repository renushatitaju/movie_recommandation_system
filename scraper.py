"""Web scraping: find a movie poster on Wikipedia (requests + BeautifulSoup)."""
import requests
from bs4 import BeautifulSoup

BASE = "https://en.wikipedia.org"
HEADERS = {"User-Agent": "CineMatchStudentProject/1.0 (educational use)"}
_cache = {}   # title -> image bytes, so we never scrape the same movie twice


def _get_page(url, params=None):
    response = requests.get(url, params=params, headers=HEADERS, timeout=8)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def find_poster_url(title):
    """Search Wikipedia, open the movie page, and read the image from its infobox."""
    query = f"{title} film"
    soup = _get_page(f"{BASE}/w/index.php", params={"search": query, "go": "Go"})

    if soup.find("table", class_="infobox") is None:       # landed on search results
        first = soup.select_one("div.mw-search-result-heading a")
        if first is None:
            return None
        soup = _get_page(BASE + first["href"])

    image = soup.select_one("table.infobox img")
    if image is None:
        return None
    src = image["src"]
    return "https:" + src if src.startswith("//") else src


def get_poster_bytes(title):
    """Return the poster image as bytes, or None if it could not be found."""
    if title in _cache:
        return _cache[title]
    try:
        url = find_poster_url(title)
        if url is None:
            return None
        response = requests.get(url, headers=HEADERS, timeout=8)
        response.raise_for_status()
        _cache[title] = response.content
        return response.content
    except requests.RequestException:
        return None          # no internet / blocked / not found


if __name__ == "__main__":
    print(find_poster_url("Avatar"))