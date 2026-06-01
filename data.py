"""Mock company research data for demo runs (no external API required)."""

import re

COMPANIES: dict[str, dict] = {
    "apple": {
        "name": "Apple Inc.",
        "aliases": ["apple", "aapl"],
        "ticker": "AAPL",
        "description": (
            "Apple designs and sells consumer electronics, software, and services "
            "including the iPhone, Mac, iPad, Apple Watch, and services like the App Store and iCloud."
        ),
        "stock_price_last_year": "$192.53 (Dec 2024 close, approximate)",
        "recent_news": "Released iPhone 16 lineup; expanding Apple Intelligence features across devices.",
        "competitors": "Samsung, Google (Pixel/Android), Microsoft (Surface), and other consumer electronics firms.",
    },
    "tesla": {
        "name": "Tesla, Inc.",
        "aliases": ["tesla", "tsla"],
        "ticker": "TSLA",
        "description": (
            "Tesla designs, manufactures, and sells electric vehicles, energy storage, "
            "and solar products. It also develops autonomous driving software."
        ),
        "stock_price_last_year": "$248.48 (Dec 2024 close, approximate)",
        "recent_news": "Continued EV production scaling; Cybertruck deliveries and FSD updates in focus.",
        "competitors": "Ford, GM, Rivian, BYD, and legacy automakers pivoting to EVs.",
    },
    "nvidia": {
        "name": "NVIDIA Corporation",
        "aliases": ["nvidia", "nvda"],
        "ticker": "NVDA",
        "description": (
            "NVIDIA designs GPUs and AI accelerators for gaming, data centers, and AI workloads. "
            "Its CUDA platform powers much of modern machine learning."
        ),
        "stock_price_last_year": "$134.29 (Dec 2024 close, approximate)",
        "recent_news": "Strong demand for AI data-center GPUs; Blackwell architecture rollout.",
        "competitors": "AMD, Intel, and cloud providers building custom AI chips.",
    },
    "google": {
        "name": "Alphabet Inc. (Google)",
        "aliases": ["google", "alphabet", "goog", "googl"],
        "ticker": "GOOGL",
        "description": (
            "Alphabet is Google's parent company. Google operates search, YouTube, Android, "
            "Google Cloud, and AI products like Gemini."
        ),
        "stock_price_last_year": "$171.86 (Dec 2024 close, approximate)",
        "recent_news": "Gemini AI integration across Search and Workspace; cloud revenue growth.",
        "competitors": "Microsoft (Bing/Copilot), Apple, Meta, and Amazon in search/ads/cloud.",
    },
    "microsoft": {
        "name": "Microsoft Corporation",
        "aliases": ["microsoft", "msft"],
        "ticker": "MSFT",
        "description": (
            "Microsoft develops software (Windows, Office), cloud services (Azure), "
            "gaming (Xbox), and AI tools including Copilot."
        ),
        "stock_price_last_year": "$421.50 (Dec 2024 close, approximate)",
        "recent_news": "Azure AI growth; Copilot expansion across enterprise products.",
        "competitors": "Google Cloud, Amazon AWS, Oracle, and Salesforce in enterprise software/cloud.",
    },
}

# Queries containing these keywords need a deeper second research pass (demo: validation loop)
PARTIAL_RESEARCH_TRIGGERS = ("stock price last year", "price last year", "last year's stock")


def resolve_company(text: str, fallback: str | None = None) -> str | None:
    """Match a company key from free text using whole-word matching."""
    lower = text.lower()
    for key, info in COMPANIES.items():
        for token in [key, *info["aliases"]]:
            if re.search(rf"\b{re.escape(token)}\b", lower):
                return key
    return fallback


def search_company(
    company_key: str,
    query: str,
    intent: str | None = None,
    attempt: int = 1,
) -> tuple[str, float]:
    """
    Mock search tool returning findings and a confidence score (0–10).

    First attempt on stock-price questions returns partial data (low confidence)
    to demonstrate the validator → research loop.
    """
    info = COMPANIES[company_key]
    lower_query = query.lower()
    needs_stock = any(t in lower_query for t in PARTIAL_RESEARCH_TRIGGERS)

    if needs_stock and attempt == 1:
        findings = (
            f"Partial data for {info['name']} ({info['ticker']}): "
            f"Historical pricing data was found but year-end figure is not yet confirmed."
        )
        return findings, 4.0

    sections: list[str] = [f"Company: {info['name']} ({info['ticker']})"]

    if intent:
        if intent == "ticker":
            sections.append(f"Ticker symbol: {info['ticker']}")
        elif intent == "business_overview":
            sections.append(f"Overview: {info['description']}")
        elif intent == "stock_price":
            sections.append(f"Stock price (last year): {info['stock_price_last_year']}")
        elif intent == "recent_news":
            sections.append(f"Recent news: {info['recent_news']}")
        elif intent == "competitors":
            sections.append(f"Key competitors: {info['competitors']}")
        elif intent == "financials":
            sections.append(f"Stock info: {info['stock_price_last_year']}")
            sections.append(f"Recent news: {info['recent_news']}")
    else:
        if any(w in lower_query for w in ("ticker", "symbol", "stock symbol")):
            sections.append(f"Ticker symbol: {info['ticker']}")
        if any(w in lower_query for w in ("do", "what is", "business", "description")):
            sections.append(f"Overview: {info['description']}")
        if needs_stock or "stock" in lower_query or "price" in lower_query:
            sections.append(f"Stock price (last year): {info['stock_price_last_year']}")
        if any(w in lower_query for w in ("news", "recent", "development", "developments")):
            sections.append(f"Recent news: {info['recent_news']}")
        if "competitor" in lower_query:
            sections.append(f"Key competitors: {info['competitors']}")

    # Default: return full profile when query is broad
    if len(sections) == 1:
        sections.extend(
            [
                f"Overview: {info['description']}",
                f"Ticker: {info['ticker']}",
                f"Stock price (last year): {info['stock_price_last_year']}",
                f"Recent news: {info['recent_news']}",
            ]
        )

    findings = "\n".join(sections)
    confidence = 8.5 if attempt > 1 and needs_stock else 7.5
    return findings, confidence
