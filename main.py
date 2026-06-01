"""
Runnable demo for the LangGraph multi-agent research assistant.

Demonstrates clarity intent routing, interrupts, validation loop, and multi-turn memory.
"""

from langgraph.types import Command

from graph import build_graph


def run_query(
    graph,
    thread_id: str,
    query: str,
    resume: str | None = None,
    *,
    show_state: bool = False,
) -> tuple[str | None, dict | None]:
    """Invoke the graph for one user turn; returns (answer, final_state)."""
    config = {"configurable": {"thread_id": thread_id}}

    if resume is not None:
        result = graph.invoke(Command(resume=resume), config)
    else:
        result = graph.invoke(
            {
                "user_query": query,
                "research_attempts": 0,
                "is_company_research": False,
                "needs_clarification": False,
            },
            config,
        )

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value if result["__interrupt__"] else {}
        print(f"\n  [INTERRUPT] {payload.get('message', 'Please clarify.')}")
        if show_state:
            print(f"  intent={result.get('intent')} needs_clarification={result.get('needs_clarification')}")
        return None, result

    if show_state:
        print(
            f"  [STATE] intent={result.get('intent')} "
            f"company={result.get('company')} "
            f"is_company_research={result.get('is_company_research')} "
            f"out_of_scope={bool(result.get('out_of_scope_reason'))}"
        )

    return result.get("final_answer"), result


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    graph = build_graph()

    # ── Intent / entity edge cases (interviewer concern) ─────────────────────
    print_section('Demo 1: "What is the stock price of Apple?" -> Apple Inc. research')
    answer, _ = run_query(
        graph, "demo-apple-stock", "What is the stock price of Apple?", show_state=True
    )
    print(answer)

    print_section('Demo 2: "What is the color of the apple?" -> out_of_scope (no research)')
    answer, _ = run_query(
        graph, "demo-apple-fruit", "What is the color of the apple?", show_state=True
    )
    print(answer)

    print_section('Demo 3: "What does Tesla do?" -> Tesla research path')
    answer, _ = run_query(graph, "demo-tesla", "What does Tesla do?", show_state=True)
    print(answer)

    print_section('Demo 4: Follow-up "What about competitors?" reuses Tesla from memory')
    run_query(graph, "demo-tesla-followup", "What does Tesla do?")
    answer, _ = run_query(
        graph, "demo-tesla-followup", "What about competitors?", show_state=True
    )
    print(answer)

    print_section('Demo 5: "What is the stock price?" -> interrupt (which company?)')
    answer, _ = run_query(graph, "demo-no-company", "What is the stock price?", show_state=True)
    if answer is None:
        answer, _ = run_query(
            graph, "demo-no-company", "What is the stock price?", resume="Microsoft", show_state=True
        )
    print(answer)

    # ── Original workflow demos ──────────────────────────────────────────────
    print_section("Demo 6: Vague pronoun query -> clarification interrupt")
    answer, _ = run_query(
        graph, "demo-vague", "What was their stock price last year?", show_state=True
    )
    if answer is None:
        answer, _ = run_query(
            graph, "demo-vague", "What was their stock price last year?", resume="Google"
        )
    print(answer)

    print_section("Demo 7: Low-confidence validation loop (Google stock last year)")
    answer, _ = run_query(
        graph, "demo-loop", "What was Google's stock price last year?", show_state=True
    )
    print(answer)

    print_section("Demo 8: Follow-up ticker question reuses Nvidia context")
    run_query(graph, "demo-followup", "What does Nvidia do?")
    answer, _ = run_query(
        graph, "demo-followup", "What is the ticker symbol?", show_state=True
    )
    print(answer)

    print("\n" + "=" * 60)
    print("All demos complete. Run tests: pytest tests/")
    print("=" * 60)


if __name__ == "__main__":
    main()
