"""Node functions for each specialized agent in the research graph."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from classifier import classifier
from data import resolve_company, search_company
from state import ResearchState


def _clarity_fields(decision, query: str) -> dict:
    """Map ClarityDecision dataclass to state update dict."""
    update: dict = {
        "user_query": query,
        "intent": decision.intent,
        "is_company_research": decision.is_company_research,
        "needs_clarification": decision.needs_clarification,
        "clarification_question": decision.clarification_question,
        "out_of_scope_reason": decision.out_of_scope_reason,
        "clarity_status": decision.clarity_status,
        "messages": [HumanMessage(content=query)],
    }
    # Only set company when resolved; omit on out-of-scope to preserve thread memory
    if decision.company is not None:
        update["company"] = decision.company
    return update


def clarity_agent(state: ResearchState) -> dict:
    """
    Clarity Agent: entity detection -> intent classification -> routing decision.

    Does NOT route on entity alone. Example:
      - "stock price of Apple"  -> Apple Inc. research
      - "color of the apple"  -> out_of_scope (fruit, not company)
    """
    query = state["user_query"].strip()
    prior_company = state.get("company")

    # Step 1-3: deterministic classifier (see classifier.py)
    decision = classifier.decide(query, prior_company=prior_company)

    # Out-of-scope: skip research entirely
    if decision.out_of_scope_reason:
        return _clarity_fields(decision, query)

    # Needs clarification: interrupt for human input (HITL)
    if decision.needs_clarification:
        clarified = interrupt(
            {
                "message": decision.clarification_question,
                "original_query": query,
            }
        )
        clarified = str(clarified).strip()
        company = resolve_company(clarified, fallback=None)

        # User may reply with just a company name; keep original question intent
        if company and len(clarified.split()) <= 2:
            decision = classifier.decide(query, prior_company=company)
        else:
            decision = classifier.decide(clarified, prior_company=prior_company)

        if decision.needs_clarification or not decision.company:
            return _clarity_fields(decision, query)

    return _clarity_fields(decision, query)


def research_agent(state: ResearchState) -> dict:
    """
    Research Agent: gather company information via mock search tool.

    Uses classified intent to focus mock search results.
    """
    company = state["company"]
    query = state["user_query"]
    intent = state.get("intent")
    attempt = state.get("research_attempts", 0) + 1

    if not company:
        return {
            "research_findings": "No company identified.",
            "confidence_score": 0.0,
            "research_attempts": attempt,
        }

    findings, confidence = search_company(company, query, intent=intent, attempt=attempt)

    return {
        "research_findings": findings,
        "confidence_score": confidence,
        "research_attempts": attempt,
    }


def validator_agent(state: ResearchState) -> dict:
    """Validator Agent: check whether research is complete enough to answer the query."""
    findings = state.get("research_findings") or ""
    confidence = state.get("confidence_score") or 0.0
    intent = state.get("intent") or ""
    query = state["user_query"].lower()

    insufficient = confidence < 6.0 or "partial data" in findings.lower()

    if intent in ("stock_price", "ticker") or any(w in query for w in ("stock", "price", "ticker")):
        if "stock price" not in findings.lower() and "ticker" not in findings.lower():
            insufficient = True

    if intent == "competitors" and "competitor" not in findings.lower():
        insufficient = True

    result = "insufficient" if insufficient else "sufficient"
    return {"validation_result": result}


def synthesis_agent(state: ResearchState) -> dict:
    """
    Synthesis Agent: format research or out-of-scope responses.

    Reads conversation history via messages for multi-turn context.
    """
    query = state["user_query"]
    out_of_scope = state.get("out_of_scope_reason")

    if out_of_scope:
        answer = (
            f"**Question:** {query}\n\n"
            f"I can help with company research (stock, news, financials, competitors). "
            f"{out_of_scope}"
        )
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    company_key = state.get("company") or "unknown"
    findings = state.get("research_findings") or "No findings available."
    intent = state.get("intent") or "general"
    attempts = state.get("research_attempts", 1)

    answer_lines = [
        f"**Question:** {query}",
        f"**Company:** {company_key.title()}",
        f"**Intent:** {intent}",
        "",
        findings,
    ]

    if attempts > 1:
        answer_lines.append(f"\n_(Research refined after {attempts} attempts.)_")

    answer = "\n".join(answer_lines)
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
