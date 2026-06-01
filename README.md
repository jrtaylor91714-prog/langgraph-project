# LangGraph Multi-Agent Research Assistant

## 1. Overview

A multi-agent **company research assistant** built with LangGraph. Users ask questions about public companies; four specialized agents collaborate to clarify intent, gather mock research, validate completeness, and synthesize a final answer.

Supported companies: **Apple**, **Tesla**, **Nvidia**, **Google/Alphabet**, **Microsoft**.

No external APIs required — research uses deterministic mock data in `data.py`.

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python main.py
pytest tests/
```

## 2. Architecture

```
START
  │
  ▼
Clarity Agent ──(out_of_scope)─────────────────────────────┐
  │                                                         │
  ├──(interrupt)──> pause for human clarification           │
  │                                                         │
  └──(clear + company)──> Research Agent                    │
                              │                             │
                    (confidence >= 6)                       │
                              │                             │
                              ▼                             │
                         Synthesis Agent <──────────────────┤
                              │              (out_of_scope) │
                    (confidence < 6)                        │
                              │                             │
                              ▼                             │
                         Validator Agent ──(retry if attempts < 3)──> Research
                              │
                              └──(sufficient OR max attempts)──> Synthesis
                                                                    │
                                                                    ▼
                                                                   END
```

| Agent | Responsibility |
|-------|----------------|
| **Clarity** | Entity detection, intent classification, routing decision |
| **Research** | Mock search, structured findings, confidence score, increment `attempts` |
| **Validator** | Check if findings answer the question; trigger retry loop |
| **Synthesis** | User-friendly `final_response` with confidence and caveats |

## 3. State schema

`ResearchState` (TypedDict in `state.py`):

| Field | Description |
|-------|-------------|
| `messages` | Conversation history (multi-turn) |
| `user_query` | Current question |
| `company` | Resolved company key |
| `intent` | Classified intent (see below) |
| `is_company_research` | Whether to route to Research |
| `clarity_status` | `"clear"` or `"needs_clarification"` |
| `needs_clarification` | Whether interrupt was triggered |
| `clarification_question` | Targeted question for user |
| `out_of_scope_reason` | Why query is outside company research |
| `research_findings` | Structured mock research output |
| `confidence_score` | 0–10 from Research Agent |
| `validation_result` | `"sufficient"` or `"insufficient"` |
| `attempts` | Research attempt counter (max 3) |
| `final_response` | Synthesis Agent output |

**Supported intents:** `stock_price`, `ticker`, `business_overview`, `recent_news`, `competitors`, `financials`, `ceo`, `products`, `out_of_scope`, `needs_clarification`

## 4. Routing logic

Three conditional routing functions in `graph.py`:

### After Clarity (`route_after_clarity`)
- `out_of_scope_reason` set → **Synthesis** (polite decline, no research)
- `is_company_research` and `company` → **Research**
- else → **END** (unresolved after interrupt)

### After Research (`route_after_research`)
- `confidence_score >= 6` → **Synthesis**
- else → **Validator**

### After Validator (`route_after_validator`)
- `validation_result == "insufficient"` AND `attempts < 3` → **Research** (loop)
- else → **Synthesis** (includes max-attempt caveat if still insufficient)

### Clarity pipeline (`classifier.py`)

Three **separate** steps — entity alone does not route:

1. **Entity detection** — is `Apple`, `AAPL`, `Tesla`, etc. mentioned?
2. **Intent classification** — business keywords vs out-of-scope (fruit/general)?
3. **Routing decision** — combine entity + intent + prior `company` from memory

| Query | Route |
|-------|-------|
| Known company + business intent | `clarity_status="clear"` → Research |
| Company-like term + fruit/general intent | out_of_scope → Synthesis |
| Business intent, no company, no memory | `interrupt()` |
| Follow-up with business intent, no company | reuse prior `company` |

## 5. How to run

```bash
pip install -r requirements.txt
python main.py      # 6 demos with [ROUTE] and [DEBUG] output
pytest tests/       # classifier unit tests
```

Interactive example:

```python
from langgraph.types import Command
from graph import build_graph

graph = build_graph()
config = {"configurable": {"thread_id": "session-1"}}

result = graph.invoke(
    {"user_query": "What does Tesla do?", "attempts": 0,
     "is_company_research": False, "needs_clarification": False},
    config,
)
print(result["final_response"])

# Follow-up (reuses Tesla from memory)
result = graph.invoke(
    {"user_query": "What about competitors?", "attempts": 0,
     "is_company_research": False, "needs_clarification": False},
    config,
)

# After interrupt
result = graph.invoke(Command(resume="Microsoft"), config)
```

## 6. Example conversations

**Demo 1 — Apple stock (company research)**
```
User: What is the stock price of Apple?
Route: clarity -> research -> synthesis
Intent: stock_price | Company: apple
```

**Demo 2 — Apple fruit (out of scope)**
```
User: What is the color of the apple?
Route: clarity -> synthesis
Intent: out_of_scope | is_company_research: False
```

**Demo 3 — Tesla overview**
```
User: What does Tesla do?
Route: clarity -> research -> synthesis
Intent: business_overview
```

**Demo 4 — Follow-up competitors**
```
Turn 1: What does Tesla do?        -> sets company=tesla
Turn 2: What about competitors?    -> reuses tesla, intent=competitors
Route: clarity -> research -> synthesis
```

**Demo 5 — Missing company (interrupt)**
```
User: What is the stock price?
Route: clarity -> INTERRUPT
Resume: Microsoft
Route: clarity -> research -> synthesis
```

**Demo 6 — Validation loop**
```
User: What was Google's stock price last year?
Route: clarity -> research -> validator -> research -> synthesis
Attempt 1: confidence=4 (partial) -> insufficient
Attempt 2: confidence=8.5 -> sufficient
```

## 7. Assumptions

- Mock data replaces Tavily/live APIs; swap `search_company()` in `data.py` for real search.
- Intent classifier is **deterministic** (keyword-based) for reliable interview evaluation.
- Whole-word entity matching (`\bapple\b`) avoids false positives like "pineapple".
- Out-of-scope responses do **not** clear prior `company` from thread memory.
- Max **3** research attempts prevents infinite loops.
- Stock-price "last year" queries simulate incomplete first-pass data to demo the validator loop.

## 8. Beyond Expected Deliverable

- **Interviewer edge-case fix:** Apple fruit vs Apple Inc. — entity detection and intent classification are separate; `"color of the apple"` never routes to Research.
- **Intent classification before routing:** `ClarityClassifier` in `classifier.py` with 10 typed intents.
- **Out-of-scope handling:** dedicated Synthesis path with polite decline (no fake company data).
- **Follow-up company reuse:** `MemorySaver` + `thread_id` + `prior_company` in clarity decision.
- **Deterministic mock data:** predictable confidence scores and partial-data paths for evaluation.
- **`ClarityClassifier` class** with dataclass `ClarityDecision` for clean separation of concerns.
- **pytest suite** (`tests/test_classifier.py`) for entity, intent, and routing tests.
- **Route/debug logging** in `main.py` via graph stream (`[ROUTE]`, `[DEBUG]` lines).
- **Structured research findings** (`company:`, `ticker:`, etc.) and CEO/products mock fields.

## Project layout

```
main.py         Demo runner (6 required scenarios)
state.py        TypedDict state schema
classifier.py   Entity + intent + routing (ClarityClassifier)
data.py         Mock company data + search tool
agents.py       Four agent node functions
graph.py        StateGraph, conditional edges, MemorySaver
tests/          pytest coverage
requirements.txt
README.md
prompt_log.txt
```
