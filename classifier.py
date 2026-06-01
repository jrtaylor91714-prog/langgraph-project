"""
Deterministic clarity pipeline: entity detection -> intent classification -> routing.

Separates three concerns so ambiguous tokens like "apple" are not treated as Apple Inc.
unless the query also signals business/company research intent.
"""

import re
from dataclasses import dataclass
from typing import Literal

from data import COMPANIES

Intent = Literal[
    "stock_price",
    "ticker",
    "business_overview",
    "recent_news",
    "competitors",
    "financials",
    "ceo",
    "products",
    "out_of_scope",
    "needs_clarification",
]

BUSINESS_KEYWORDS = (
    "stock",
    "price",
    "ticker",
    "symbol",
    "shares",
    "revenue",
    "market cap",
    "earnings",
    "financials",
    "business",
    "company",
    "competitors",
    "competitor",
    "news",
    "ceo",
    "products",
    "trading",
    "ytd",
    "market",
)

OUT_OF_SCOPE_KEYWORDS = (
    "color",
    "taste",
    "recipe",
    "fruit",
    "eat",
    "nutrition",
    "calories",
    "tree",
    "ripe",
    "orchard",
)

FOLLOW_UP_PATTERNS = (
    "what about",
    "how about",
    "tell me more",
    "their ",
    "its ",
    "the company",
    "that company",
)


@dataclass
class ClarityDecision:
    intent: Intent
    is_company_research: bool
    needs_clarification: bool
    clarification_question: str | None
    out_of_scope_reason: str | None
    company: str | None
    clarity_status: Literal["clear", "needs_clarification"]


class ClarityClassifier:
    """
    1. detect_entity   - is a company/ticker mentioned?
    2. classify_intent - is this a business research question?
    3. decide          - research, interrupt, or out-of-scope?
    """

    def detect_entity(self, text: str) -> str | None:
        """Return a known company key if name/ticker appears as a whole word."""
        lower = text.lower()
        for key, info in COMPANIES.items():
            for token in [key, *info["aliases"]]:
                if re.search(rf"\b{re.escape(token)}\b", lower):
                    return key
        return None

    def _has_business_signal(self, text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in BUSINESS_KEYWORDS) or any(
            p in lower for p in FOLLOW_UP_PATTERNS
        )

    def _has_out_of_scope_signal(self, text: str) -> bool:
        return any(kw in text.lower() for kw in OUT_OF_SCOPE_KEYWORDS)

    def classify_intent(self, text: str, entity: str | None) -> Intent:
        """Keyword-based intent; out-of-scope beats company entity when non-business."""
        lower = text.lower()
        business = self._has_business_signal(text)
        out_of_scope = self._has_out_of_scope_signal(text)

        if out_of_scope and not business:
            return "out_of_scope"
        if entity and out_of_scope and not business:
            return "out_of_scope"

        if any(w in lower for w in ("competitor", "competitors", "rival", "rivals")):
            return "competitors"
        if any(w in lower for w in ("ticker", "symbol", "stock symbol")):
            return "ticker"
        if any(w in lower for w in ("stock", "price", "trading", "shares")):
            return "stock_price"
        if any(w in lower for w in ("financials", "earnings", "revenue", "market cap")):
            return "financials"
        if any(w in lower for w in ("news", "recent", "development", "developments")):
            return "recent_news"
        if "ceo" in lower:
            return "ceo"
        if "product" in lower:
            return "products"
        if any(w in lower for w in ("do", "what is", "business", "overview", "about")):
            return "business_overview"

        if business and not entity:
            return "needs_clarification"
        if entity and business:
            return "business_overview"
        if entity and not business and not out_of_scope:
            return "business_overview"

        return "needs_clarification"

    def decide(self, query: str, prior_company: str | None = None) -> ClarityDecision:
        entity = self.detect_entity(query)
        intent = self.classify_intent(query, entity)
        business = self._has_business_signal(query)

        company = entity
        needs_clarification = False
        clarification_question: str | None = None
        out_of_scope_reason: str | None = None

        if intent == "out_of_scope":
            if entity:
                out_of_scope_reason = (
                    f'Your question mentions "{entity.title()}" but appears to be '
                    "non-business (e.g. fruit/general knowledge), not company research."
                )
            else:
                out_of_scope_reason = (
                    "This question is outside company research scope "
                    "(e.g. recipes, nutrition, general trivia)."
                )
            return ClarityDecision(
                intent=intent,
                is_company_research=False,
                needs_clarification=False,
                clarification_question=None,
                out_of_scope_reason=out_of_scope_reason,
                company=None,
                clarity_status="needs_clarification",
            )

        if business and not company:
            company = prior_company

        if business and not company:
            needs_clarification = True
            intent = "needs_clarification"
            if "stock" in query.lower() or "price" in query.lower():
                clarification_question = "Which company's stock price would you like to know?"
            elif "competitor" in query.lower():
                clarification_question = "Which company's competitors are you asking about?"
            elif "ticker" in query.lower() or "symbol" in query.lower():
                clarification_question = "Which company's ticker symbol do you need?"
            else:
                clarification_question = "Which company are you asking about?"

        if needs_clarification:
            return ClarityDecision(
                intent=intent,
                is_company_research=False,
                needs_clarification=True,
                clarification_question=clarification_question,
                out_of_scope_reason=None,
                company=None,
                clarity_status="needs_clarification",
            )

        return ClarityDecision(
            intent=intent if intent != "needs_clarification" else "business_overview",
            is_company_research=True,
            needs_clarification=False,
            clarification_question=None,
            out_of_scope_reason=None,
            company=company,
            clarity_status="clear",
        )


classifier = ClarityClassifier()
