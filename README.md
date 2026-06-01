# LangGraph Multi-Agent Research Assistant

A runnable Python exercise: four specialized agents (Clarity, Research, Validator, Synthesis) orchestrated with LangGraph StateGraph, conditional routing, human-in-the-loop interrupts, and multi-turn memory.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python main.py                  # 8 demo scenarios
pytest tests/                   # classifier + routing tests
```

No API keys required. Research uses mock data in `data.py`.

## Project layout

```
main.py         Demo runner
state.py        TypedDict state schema
classifier.py   Entity detection + intent classification + routing (ClarityClassifier)
data.py         Mock company data + search tool
agents.py       Four agent node functions
graph.py        StateGraph, 3 conditional routers, MemorySaver
tests/          pytest coverage for clarity edge cases
```

## Agent architecture

| Agent | Output | Routes to |
|-------|--------|-----------|
| **Clarity** | `intent`, `is_company_research`, `clarity_status`, etc. | Research, Synthesis (out-of-scope), or `interrupt()` |
| **Research** | `research_findings`, `confidence_score` | Synthesis (conf >= 6) or Validator |
| **Validator** | `validation_result` | Research loop (attempts < 3) or Synthesis |
| **Synthesis** | `final_answer` | END |

## Clarity pipeline (Apple fruit vs Apple Inc.)

The Clarity Agent uses **three separate steps** (see `classifier.py`):

1. **Entity detection** — finds company/ticker tokens (`apple`, `TSLA`) without routing.
2. **Intent classification** — business keywords (stock, ticker, news) vs out-of-scope (color, fruit, recipe).
3. **Routing decision** — combines entity + intent + prior `company` from memory.

| Query | Entity | Intent | Route |
|-------|--------|--------|-------|
| "What is the stock price of Apple?" | apple | stock_price | Research (Apple Inc.) |
| "What is the color of the apple?" | apple | out_of_scope | Synthesis (decline, no research) |
| "What does Tesla do?" | tesla | business_overview | Research |
| "What about competitors?" (after Tesla) | — | competitors | Research (reuses Tesla) |
| "What is the stock price?" | — | needs_clarification | Interrupt |

## State schema

Required fields in `ResearchState`:

- `messages`, `user_query`, `company`
- `intent`, `is_company_research`, `needs_clarification`, `clarification_question`, `out_of_scope_reason`
- `clarity_status`, `research_findings`, `confidence_score`, `validation_result`, `research_attempts`, `final_answer`

## Deliverable checklist

| Requirement | Status |
|-------------|--------|
| 4 specialized agents | Done |
| TypedDict state with all fields | Done |
| 3 conditional routing functions | Done (`route_after_clarity`, `route_after_research`, `route_after_validator`) |
| Validator -> Research loop with attempt counter | Done (max 3) |
| Interrupt for unclear queries | Done |
| Multi-turn memory (`MemorySaver` + `thread_id`) | Done |
| 2+ example conversation turns | Done (8 demos in `main.py`) |
| README with run instructions | Done |
| Intent classification / Apple edge case | Done |

## Assumptions

- Mock search replaces Tavily/API calls; structure allows swapping in a real search tool in `research_agent`.
- Supported companies: Apple, Tesla, Nvidia, Google/Alphabet, Microsoft.
- Intent classifier is deterministic (keyword-based), not LLM-based, for predictable interview demos.
- Stock-price questions simulate a low-confidence first pass to demonstrate the validation loop.
- Out-of-scope queries receive a polite decline via Synthesis without clearing prior `company` context.

## Beyond Expected Deliverable

- **`ClarityClassifier` class** separating entity detection, intent classification, and routing decision.
- **`classifier.py` module** with typed intents and targeted clarification questions.
- **pytest suite** (`tests/test_classifier.py`) covering Apple Inc. vs apple fruit and follow-up routing.
- **Competitors intent** with mock competitor data per company.
- **Demo state logging** in `main.py` showing `intent`, `company`, and routing flags per scenario.
- **Whole-word entity matching** (`\bapple\b`) to reduce false positives (e.g. "pineapple").

## Interactive usage

```python
from langgraph.types import Command
from graph import build_graph

graph = build_graph()
config = {"configurable": {"thread_id": "session-1"}}

result = graph.invoke(
    {"user_query": "What does Tesla do?", "research_attempts": 0,
     "is_company_research": False, "needs_clarification": False},
    config,
)
print(result["final_answer"])

# Follow-up in same thread
result = graph.invoke(
    {"user_query": "What about competitors?", "research_attempts": 0,
     "is_company_research": False, "needs_clarification": False},
    config,
)

# After interrupt
result = graph.invoke(Command(resume="Microsoft"), config)
```
