"""LangGraph StateGraph assembly with conditional routing and MemorySaver."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from state import CONFIDENCE_THRESHOLD, MAX_ATTEMPTS, ResearchState


def route_after_clarity(state: ResearchState) -> str:
    """Route on routing decision (not entity alone)."""
    if state.get("out_of_scope_reason"):
        return "synthesis"
    if state.get("is_company_research") and state.get("company"):
        return "research"
    return END


def route_after_research(state: ResearchState) -> str:
    """High confidence -> synthesis; low -> validator."""
    if (state.get("confidence_score") or 0.0) >= CONFIDENCE_THRESHOLD:
        return "synthesis"
    return "validator"


def route_after_validator(state: ResearchState) -> str:
    """Loop research if insufficient and attempts remain; else synthesize."""
    attempts = state.get("attempts") or 0
    if state.get("validation_result") == "insufficient" and attempts < MAX_ATTEMPTS:
        return "research"
    return "synthesis"


def build_graph():
    """
    Graph flow:

        START -> clarity --(research)--> research --(conf >= 6)--> synthesis -> END
                    |                       |
              (out_of_scope)            (conf < 6)
                    |                       |
                    +------> synthesis <----+
                            validator --(retry, attempts < 3)--+
                    (interrupt / unclear) -> END
    """
    from agents import clarity_agent, research_agent, synthesis_agent, validator_agent

    builder = StateGraph(ResearchState)

    builder.add_node("clarity", clarity_agent)
    builder.add_node("research", research_agent)
    builder.add_node("validator", validator_agent)
    builder.add_node("synthesis", synthesis_agent)

    builder.set_entry_point("clarity")

    builder.add_conditional_edges(
        "clarity",
        route_after_clarity,
        {"research": "research", "synthesis": "synthesis", END: END},
    )
    builder.add_conditional_edges(
        "research",
        route_after_research,
        {"synthesis": "synthesis", "validator": "validator"},
    )
    builder.add_conditional_edges(
        "validator",
        route_after_validator,
        {"research": "research", "synthesis": "synthesis"},
    )
    builder.add_edge("synthesis", END)

    return builder.compile(checkpointer=MemorySaver())
