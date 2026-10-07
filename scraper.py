"""Find a movie poster using the Wikipedia API (requests)."""
import requests

BASE = "https://en.wikipedia.org"
HEADERS = {"User-Agent": "CineMatchStudentProject/1.0 (educational use)"}
_cache = {}   # title -> image bytes, so we never download the same poster twice


def find_poster_url(title):
    """Search Wikipedia and return the poster link of the best matching page."""
    params = {
        "action": "query", "format": "json",
        "generator": "search", "gsrsearch": f"{title} film",
        "gsrlimit": 5, "prop": "pageimages",
        "pithumbsize": 500,
        "pilicense": "any",          # allow non-free images like posters
    }
    response = requests.get(f"{BASE}/w/api.php", params=params,
                            headers=HEADERS, timeout=8)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    pages = sorted(pages.values(), key=lambda p: p["index"])

    # first choice: a page with "film" in its title, e.g. Titanic (1997 film)
    for page in pages:
        if "film" in page["title"].lower() and "thumbnail" in page:
            return page["thumbnail"]["source"]

    # second choice: first result that has any image, e.g. The Godfather
    for page in pages:
        if "thumbnail" in page:
            return page["thumbnail"]["source"]
    return None


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
    except requests.RequestException as error:
        print("Poster error:", error)      # shows the real problem in the terminal
        return None


if __name__ == "__main__":
    print(find_poster_url("Titanic"))