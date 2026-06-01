"""Shared graph state definition."""

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages

IntentType = Literal[
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


class ResearchState(TypedDict):
    """State passed between all agents in the research graph."""

    messages: Annotated[list, add_messages]
    user_query: str
    company: Optional[str]

    # Clarity Agent (entity + intent + routing)
    intent: Optional[IntentType]
    is_company_research: bool
    clarity_status: Optional[Literal["clear", "needs_clarification"]]
    needs_clarification: bool
    clarification_question: Optional[str]
    out_of_scope_reason: Optional[str]

    # Research Agent
    research_findings: Optional[str]
    confidence_score: Optional[float]
    attempts: int

    # Validator Agent
    validation_result: Optional[Literal["sufficient", "insufficient"]]

    # Synthesis Agent
    final_response: Optional[str]


MAX_ATTEMPTS = 3
CONFIDENCE_THRESHOLD = 6.0
