"""Mock company research data for demo runs (no external API required)."""

import re

COMPANIES: dict[str, dict] = {
    "apple": {
        "name": "Apple Inc.",
        "aliases": ["apple", "apple inc.", "apple inc", "aapl"],
        "ticker": "AAPL",
        "description": (
            "Apple designs and sells consumer electronics, software, and services "
            "including the iPhone, Mac, iPad, Apple Watch, and services like the App Store and iCloud."
        ),
        "stock_price_last_year": "$192.53 (Dec 2024 close, approximate)",
        "recent_news": "Released iPhone 16 lineup; expanding Apple Intelligence features across devices.",
        "competitors": "Samsung, Google (Pixel/Android), Microsoft (Surface), and other consumer electronics firms.",
        "ceo": "Tim Cook",
        "products": "iPhone, Mac, iPad, Apple Watch, AirPods, and services (App Store, iCloud).",
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
        "ceo": "Elon Musk (CEO role; also leads other ventures)",
        "products": "Model 3/Y/S/X, Cybertruck, Powerwall, Megapack, and Full Self-Driving software.",
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
        "ceo": "Jensen Huang",
        "products": "GeForce GPUs, data-center GPUs (H100/Blackwell), CUDA, and AI enterprise software.",
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
        "ceo": "Sundar Pichai (Alphabet & Google CEO)",
        "products": "Search, YouTube, Android, Google Cloud, Workspace, and Gemini AI.",
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
        "ceo": "Satya Nadella",
        "products": "Windows, Office 365, Azure, Xbox, LinkedIn, and Copilot AI.",
    },
}

# First research pass returns partial/low-confidence data (demo: validator loop)
PARTIAL_RESEARCH_TRIGGERS = (
    "stock price last year",
    "price last year",
    "last year's stock",
    "incomplete",
)


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
    Mock search tool returning structured findings and confidence_score (0-10).

    Attempt 1 on certain stock queries returns partial data (confidence < 6)
    to demonstrate Research -> Validator -> Research loop.
    """
    info = COMPANIES[company_key]
    lower_query = query.lower()
    needs_partial = any(t in lower_query for t in PARTIAL_RESEARCH_TRIGGERS)

    if needs_partial and attempt == 1:
        findings = (
            f"[partial]\n"
            f"company: {info['name']}\n"
            f"ticker: {info['ticker']}\n"
            f"note: Historical pricing found but year-end figure not yet confirmed."
        )
        return findings, 4.0

    lines = [
        f"company: {info['name']}",
        f"ticker: {info['ticker']}",
    ]

    if intent == "ticker":
        lines.append(f"ticker_symbol: {info['ticker']}")
    elif intent == "business_overview":
        lines.append(f"overview: {info['description']}")
    elif intent == "stock_price":
        lines.append(f"stock_price_last_year: {info['stock_price_last_year']}")
    elif intent == "recent_news":
        lines.append(f"recent_news: {info['recent_news']}")
    elif intent == "competitors":
        lines.append(f"competitors: {info['competitors']}")
    elif intent == "financials":
        lines.append(f"stock_info: {info['stock_price_last_year']}")
        lines.append(f"recent_news: {info['recent_news']}")
    elif intent == "ceo":
        lines.append(f"ceo: {info['ceo']}")
    elif intent == "products":
        lines.append(f"products: {info['products']}")
    else:
        if any(w in lower_query for w in ("ticker", "symbol")):
            lines.append(f"ticker_symbol: {info['ticker']}")
        if any(w in lower_query for w in ("do", "what is", "business")):
            lines.append(f"overview: {info['description']}")
        if "stock" in lower_query or "price" in lower_query:
            lines.append(f"stock_price_last_year: {info['stock_price_last_year']}")

    if len(lines) == 2:
        lines.extend(
            [
                f"overview: {info['description']}",
                f"stock_price_last_year: {info['stock_price_last_year']}",
                f"recent_news: {info['recent_news']}",
            ]
        )

    findings = "\n".join(lines)
    confidence = 8.5 if attempt > 1 and needs_partial else 7.5
    return findings, confidence
