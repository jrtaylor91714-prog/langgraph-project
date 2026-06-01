"""
Runnable demo for the LangGraph multi-agent research assistant.

Prints route/debug info for each scenario so evaluators can see which path was taken.
"""

from langgraph.types import Command

from graph import build_graph


def _initial_state(query: str) -> dict:
    return {
        "user_query": query,
        "attempts": 0,
        "is_company_research": False,
        "needs_clarification": False,
    }


def run_query(
    graph,
    thread_id: str,
    query: str,
    resume: str | None = None,
) -> tuple[str | None, dict, list[str]]:
    """
    Run one turn; returns (final_response, state, route_path).

    Uses stream updates to capture which nodes executed (for demo visibility).
    """
    config = {"configurable": {"thread_id": thread_id}}
    inp = Command(resume=resume) if resume is not None else _initial_state(query)

    route: list[str] = []
    interrupt_message: str | None = None

    for update in graph.stream(inp, config, stream_mode="updates"):
        if "__interrupt__" in update:
            route.append("clarity")
            interrupts = update["__interrupt__"]
            if interrupts:
                payload = interrupts[0].value
                if isinstance(payload, dict):
                    interrupt_message = payload.get("message")
        else:
            route.extend(update.keys())

    snapshot = graph.get_state(config)
    state = dict(snapshot.values)

    if snapshot.next:
        if not interrupt_message and snapshot.tasks:
            for task in snapshot.tasks:
                if task.interrupts:
                    payload = task.interrupts[0].value
                    if isinstance(payload, dict):
                        interrupt_message = payload.get("message")
                    break
        print(f"  [ROUTE] START -> {' -> '.join(route)} -> INTERRUPT")
        print(f"  [INTERRUPT] {interrupt_message or 'Please clarify.'}")
        _print_debug(state, route)
        return None, state, route

    print(f"  [ROUTE] START -> {' -> '.join(route)} -> END")
    _print_debug(state, route)
    return state.get("final_response"), state, route


def _print_debug(state: dict, route: list[str]) -> None:
    print(
        "  [DEBUG] "
        f"intent={state.get('intent')} | "
        f"company={state.get('company')} | "
        f"clarity_status={state.get('clarity_status')} | "
        f"is_company_research={state.get('is_company_research')} | "
        f"out_of_scope={bool(state.get('out_of_scope_reason'))} | "
        f"attempts={state.get('attempts')} | "
        f"confidence={state.get('confidence_score')} | "
        f"validation={state.get('validation_result')} | "
        f"nodes={route}"
    )


def print_section(num: int, title: str, expected: str) -> None:
    print("\n" + "=" * 70)
    print(f"Demo {num}: {title}")
    print(f"Expected: {expected}")
    print("=" * 70)


def main() -> None:
    graph = build_graph()

    print_section(
        1,
        '"What is the stock price of Apple?"',
        "Apple Inc., stock_price intent, clarity -> research -> synthesis",
    )
    answer, _, _ = run_query(graph, "d1", "What is the stock price of Apple?")
    print("\n", answer)

    print_section(
        2,
        '"What is the color of the apple?"',
        "out_of_scope, clarity -> synthesis (no research)",
    )
    answer, _, _ = run_query(graph, "d2", "What is the color of the apple?")
    print("\n", answer)

    print_section(
        3,
        '"What does Tesla do?"',
        "Tesla, business_overview, clarity -> research -> synthesis",
    )
    answer, _, _ = run_query(graph, "d3", "What does Tesla do?")
    print("\n", answer)

    print_section(
        4,
        'Follow-up: "What about competitors?" (after Tesla)',
        "reuse Tesla from memory, competitors intent, research path",
    )
    run_query(graph, "d4", "What does Tesla do?")
    answer, _, _ = run_query(graph, "d4", "What about competitors?")
    print("\n", answer)

    print_section(
        5,
        '"What is the stock price?"',
        "interrupt asking which company, then research after resume",
    )
    answer, _, _ = run_query(graph, "d5", "What is the stock price?")
    if answer is None:
        answer, _, _ = run_query(graph, "d5", "What is the stock price?", resume="Microsoft")
    print("\n", answer)

    print_section(
        6,
        '"What was Google\'s stock price last year?"',
        "low confidence attempt 1 -> validator -> research loop -> synthesis",
    )
    answer, state, route = run_query(graph, "d6", "What was Google's stock price last year?")
    print("\n", answer)
    if state.get("attempts", 0) >= 2 and "validator" in route:
        print("  [OK] Validation loop demonstrated.")

    print("\n" + "=" * 70)
    print("All demos complete. Run tests: pytest tests/")
    print("=" * 70)


if __name__ == "__main__":
    main()
