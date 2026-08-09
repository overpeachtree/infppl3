from ddgs import DDGS


def search_web(query: str, max_results: int = 5):
    """
    Search the web and return structured results.
    """

    print(f"\n[TOOL] search_web")
    print(f"Query: {query}")

    raw_results = list(
        DDGS().text(
            query,
            region="us-en",
            max_results=max_results
        )
    )

    results = []

    for item in raw_results:
        if not item.get("title") or not item.get("href"):
            continue

        results.append({
            "title": item.get("title", ""),
            "url": item.get("href", ""),
            "snippet": item.get("body", "")
        })

    return results
