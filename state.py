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
    "out_of_scope",
    "needs_clarification",
]


class ResearchState(TypedDict):
    """State passed between all agents in the research graph."""

    # Conversation history (multi-turn support via thread_id + MemorySaver)
    messages: Annotated[list, add_messages]

    # Current user question
    user_query: str

    # Resolved company key (from query, clarification, or prior turn)
    company: Optional[str]

    # Clarity Agent: entity + intent + routing (three-step pipeline)
    intent: Optional[IntentType]
    is_company_research: bool
    needs_clarification: bool
    clarification_question: Optional[str]
    out_of_scope_reason: Optional[str]
    clarity_status: Optional[Literal["clear", "needs_clarification"]]

    # Research Agent output
    research_findings: Optional[str]
    confidence_score: Optional[float]

    # Validator Agent output
    validation_result: Optional[Literal["sufficient", "insufficient"]]

    # Loop counter for research <-> validator retries
    research_attempts: int

    # Synthesis Agent output
    final_answer: Optional[str]
