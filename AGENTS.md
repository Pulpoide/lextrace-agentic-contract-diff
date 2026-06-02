# AGENTS.md — LexTrace

Behavioral guidelines and project context for AI agents working in this repository.
Read this file fully before making any changes.

---

## Project Context

**LexTrace** is a multi-agent pipeline that detects semantic discrepancies between a legal contract and its amendment. It extracts text from scanned images via GPT-4o Vision, maps corresponding sections with a **Cartographer Agent** (semantic mapper), detects legal changes with a **Detective Agent** (anomaly detector), and returns structured JSON validated with Pydantic v2.

**Two entry points exist:**
- `streamlit run app.py` — Web UI with image upload, OCR, manual text override, and results visualization.
- `python -m src.main <path_original> <path_amendment>` — CLI mode, same pipeline.

**Observability:** Every execution generates a `lextrace-pipeline` trace in Langfuse (v3+) with a span hierarchy via `propagate_attributes()`.

---

## Stack

| Layer              | Technology                             |
|--------------------|----------------------------------------|
| Language           | Python 3.12+                           |
| Package Manager    | uv (with pyproject.toml)               |
| LLM Framework      | LangChain 0.3+                         |
| Models             | GPT-4o Vision (OCR), GPT-4o (agents)   |
| Validation         | Pydantic v2                            |
| Observability      | Langfuse v3+                           |
| UI                 | Streamlit 1.0+                         |
| CLI                | argparse (built-in)                    |
| Testing            | pytest 9.0+, pytest-cov, pytest-mock  |

---

## File Map & Roles

| File                              | Lines | Role                                                    |
|-----------------------------------|-------|---------------------------------------------------------|
| `src/pipeline.py`                 | ~50   | **PipelineOrchestrator** — Central orchestration layer  |
| `src/agents/contextualizer.py`    | 64    | **ContextualizationAgent** ("Cartographer") — Semantic mapper   |
| `src/agents/extractor.py`         | 74    | **ExtractionAgent** ("Detective") — Anomaly detector    |
| `src/models.py`                   | 42    | **Pydantic v2 schemas** — Data validation boundaries   |
| `src/utils/image_processor.py`    | 93    | **GPT-4o Vision utilities** — OCR extraction            |
| `app.py`                          | ~470  | **Streamlit interface** — Web UI entry point            |
| `src/main.py`                     | ~140  | **CLI entry point** — Terminal execution       |

---

## Commands

```bash
# Install dependencies (uv package manager)
uv sync

# Run web interface
streamlit run app.py

# Run CLI
python -m src.main <path_original_contract> <path_amendment>

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Architecture Invariants — Do Not Change Without Discussion

These decisions were made deliberately. Do not refactor them unless explicitly asked:

1. **Two-agent design:** `ContextualizationAgent` (Cartographer) maps sections first; `ExtractionAgent` (Detective) detects changes second. This is not over-engineering — it prevents LLM context fatigue. Do not collapse into a single prompt.

2. **`temperature=0` on all agents:** Legal domain requires deterministic output. Contracts must yield identical analyses for identical inputs. Do not raise this.

3. **`with_structured_output()`:** Schema is enforced at the API level (via `ChatOpenAI(...).with_structured_output()`). This returns Pydantic objects directly. Do not replace with manual parsing.

4. **`propagate_attributes()` in Langfuse:** Parent trace hierarchy is intentional for audit trails. Do not flatten spans.

5. **Pydantic v2 validation boundaries:** `ContractChangeOutput` and `SectionMapping` are the validated output boundaries. Do not bypass validation.

6. **CLI via `python -m src.main`:** Entry point follows Python module conventions. Do not change to direct script execution.

---

## Agent-Rules: Behavioral Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. **Bias toward caution over speed.**

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

**Self-check:** "Would a senior engineer call this overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd code it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

**The test:** Every changed line traces directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

**Strong success criteria** let you loop independently. **Weak criteria** ("make it work") require constant clarification.

---

## Testing Strategy

- **Fixtures:** See `tests/fixtures/` for reusable test data (e.g., minimal images for Base64 tests).
- **Test files:** `test_agents.py`, `test_image_processor.py`, `test_pipeline.py`.
- **Coverage gaps:** `src/main.py` is omitted from coverage (see `pyproject.toml`).

---

## Observability & Logging

- **Langfuse trace name:** `lextrace-pipeline` (root span).
- **Child spans:** One for each agent execution + OCR step.
- **Metadata:** Token counts, latencies, processed character counts, origin interface (CLI vs. Streamlit).

---

## Development Notes

- **Language style:** Spanish docstrings & comments (matching existing code). Agent system prompts are bilingual.
- **Error handling:** Use `RateLimitError` and `APITimeoutError` from OpenAI client (see `app.py`).
- **Environment:** Requires `.env` with `OPENAI_API_KEY`, optional `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
