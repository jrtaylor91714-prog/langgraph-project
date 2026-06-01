"""Node functions for each specialized agent in the research graph."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from classifier import classifier
from data import resolve_company, search_company
from state import MAX_ATTEMPTS, ResearchState


def _clarity_fields(decision, query: str) -> dict:
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
    if decision.company is not None:
        update["company"] = decision.company
    return update


def clarity_agent(state: ResearchState) -> dict:
    """
    Clarity Agent: entity detection -> intent classification -> routing decision.

    Does NOT route on entity alone:
      - "stock price of Apple" -> Apple Inc. research
      - "color of the apple" -> out_of_scope (no research)
    """
    query = state["user_query"].strip()
    prior_company = state.get("company")

    decision = classifier.decide(query, prior_company=prior_company)

    if decision.out_of_scope_reason:
        return _clarity_fields(decision, query)

    if decision.needs_clarification:
        clarified = interrupt(
            {
                "message": decision.clarification_question,
                "original_query": query,
            }
        )
        clarified = str(clarified).strip()
        company = resolve_company(clarified, fallback=None)

        if company and len(clarified.split()) <= 2:
            decision = classifier.decide(query, prior_company=company)
        else:
            decision = classifier.decide(clarified, prior_company=prior_company)

        if decision.needs_clarification or not decision.company:
            return _clarity_fields(decision, query)

    return _clarity_fields(decision, query)


def research_agent(state: ResearchState) -> dict:
    """Research Agent: mock search, structured findings, confidence_score, increment attempts."""
    company = state["company"]
    intent = state.get("intent")
    attempt = state.get("attempts", 0) + 1

    if not company:
        return {
            "research_findings": "error: no company identified",
            "confidence_score": 0.0,
            "attempts": attempt,
        }

    findings, confidence = search_company(
        company, state["user_query"], intent=intent, attempt=attempt
    )
    return {
        "research_findings": findings,
        "confidence_score": confidence,
        "attempts": attempt,
    }


def validator_agent(state: ResearchState) -> dict:
    """Validator Agent: check if findings answer the user's question."""
    findings = (state.get("research_findings") or "").lower()
    confidence = state.get("confidence_score") or 0.0
    intent = state.get("intent") or ""

    insufficient = confidence < 6.0 or "[partial]" in findings or "partial data" in findings

    if intent == "stock_price" and "stock_price" not in findings:
        insufficient = True
    if intent == "ticker" and "ticker" not in findings:
        insufficient = True
    if intent == "competitors" and "competitors" not in findings:
        insufficient = True
    if intent == "ceo" and "ceo" not in findings:
        insufficient = True
    if intent == "products" and "products" not in findings:
        insufficient = True

    return {"validation_result": "insufficient" if insufficient else "sufficient"}


def synthesis_agent(state: ResearchState) -> dict:
    """Synthesis Agent: user-friendly final_response with confidence and caveats."""
    query = state["user_query"]
    out_of_scope = state.get("out_of_scope_reason")

    if out_of_scope:
        answer = (
            f"Question: {query}\n\n"
            "This assistant handles company research (stock, news, financials, competitors). "
            f"{out_of_scope}"
        )
        return {"final_response": answer, "messages": [AIMessage(content=answer)]}

    company = (state.get("company") or "unknown").title()
    intent = state.get("intent") or "general"
    findings = state.get("research_findings") or "No findings available."
    confidence = state.get("confidence_score")
    attempts = state.get("attempts", 1)
    validation = state.get("validation_result")

    lines = [
        f"Question: {query}",
        f"Company: {company}",
        f"Intent: {intent}",
        "",
        "Key facts:",
        findings,
    ]

    if confidence is not None:
        lines.append(f"\nConfidence: {confidence}/10")

    if attempts > 1:
        lines.append(f"Research attempts: {attempts}")

    if validation == "insufficient" and attempts >= MAX_ATTEMPTS:
        lines.append(
            "\nCaveat: Could not fully verify all requested information after "
            f"{MAX_ATTEMPTS} research attempts. Answer may be incomplete."
        )

    answer = "\n".join(lines)
    return {"final_response": answer, "messages": [AIMessage(content=answer)]}
