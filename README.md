# Job Search Agent

Student project for **Designing LLM/VLM Agents**.

## Setup

```bash
uv sync --locked
cp .env.example .env
uv run pytest
```

Add the individual OpenRouter key supplied by the instructor to `.env`. Never
commit this file or place the key in code, traces, issues, or reports.

## Project Brief

- **Project title:** Job Search Agent for a first job.
- **Intended user:** A student or new graduate looking for an internship or a
  junior position.
- **Problem and primary task:** The user asks, for example: *"Find offers that
  fit my profile, skip the ones I already applied to, and tell me which skills
  I am missing for offers that almost match."* Doing this by hand means reading
  many offers and comparing each one with one's own skills, location, and
  salary needs.
- **Why an agent is appropriate:** The answer cannot be produced in one model
  response. The agent must read the profile, decide which searches to run, pick
  promising offers, check them against the profile, notice already-applied and
  near-miss offers, and only then answer. What it does next depends on what
  each tool returned.
- **Planned data:** `data/offers.json` (30 fictional offers, J01-J30),
  `data/profile.json` (fictional candidate), `data/applications.csv` (offers
  the candidate already applied to).
- **Planned tools:** Meeting 1: read-only `get_profile`, `search_offers`,
  `get_offer`, `check_match`, and `final_answer`. Meeting 2: a reversible write
  tool (`save_offer` to a watchlist) and a sensitive tool (`send_application`,
  a mock that needs user confirmation).
- **Planned visual input:** Screenshots of fictional job offers (Meeting 4).
- **Success criteria:** For the main example request the agent recommends
  exactly the offers that meet every hard constraint (level, location, salary,
  required skills) and were not applied to: **J01, J19, J24**. It does not
  recommend J05, J11, J29 (already applied), and it names missing skills for
  near misses such as J03 (Django), J08 (Power BI), J14 (PyTorch),
  J22 (Airflow), J26 (Google Analytics), J27 (Docker). It stays within
  12 steps and USD 0.10.
- **Non-goals and safety boundaries:** No real job boards, no real personal
  data, no sending of real applications. The model never decides whether an
  offer matches: Python does every comparison.

## How the Agent Works

```text
goal -> model chooses a tool -> Python checks and runs it
     -> result goes back to the model -> decide again or stop
```

- `src/agent_project/tools.py`: tools and their strict Pydantic argument
  models (`StrictArgs`: no extra fields, no type coercion). `tool_schemas()`
  builds the function-calling schemas from the same models. Unknown tools and
  invalid arguments return an error observation, so the model can correct
  itself.
- `src/agent_project/agent.py`: the bounded loop. It stops when:
  - `final_answer` is accepted by `check_final` (every recommended offer was
    checked with `check_match` in this run, is a full match, and was not
    applied to);
  - 3 consecutive steps fail;
  - the run costs USD 0.10 or more;
  - the step limit is reached (default 10, hard cap 12);
  - the model client raises an error.
- Identical tool calls are not executed twice (`duplicate_call`).
- Every run writes a JSONL trace to `traces/` with the goal, every model
  decision, every tool call and result, cost, and latency.

## Commands

```bash
uv run pytest                 # deterministic tests, mock client only
uv run ruff check .
uv run python -m agent_project.agent            # live run, spends credit
uv run python -m agent_project.agent "your request" --max-steps 8
uv run python -m agent_project.ui               # local web UI on 127.0.0.1:8000
```

The web UI (`src/agent_project/ui.py` and `ui.html`, standard library only)
lists traces from `traces/` and `evidence/` and shows each step: the model's
tool call, the checked result, provider, tokens, cost, and latency. Viewing
traces is free; the **Run agent** button starts a live run and the steps
appear as the trace is written.

## Lessons From Live Runs

Traces: `evidence/run-success.jsonl`, `evidence/run-step-limit.jsonl`.
Details in `evidence/REPORT.md`.

- **Too narrow search (2026-10-22).** The model filtered `search_offers` by
  `level="junior"` although the profile also accepts `intern`, and accepted
  10 of 12 results without searching again. It recommended J01 and J19 but
  missed J24. `check_final` kept the answer correct (nothing wrong was
  recommended) but cannot make it complete. Planned control: report
  truncated results and warn when filters are narrower than the profile.
- **Step limit works.** With `--max-steps 2` the run stopped after the
  search with `step_limit` and no answer.

## Course Limits

- One model for all live work: `google/gemma-4-26b-a4b-it`.
- USD 2 per key for the whole course, enforced by OpenRouter.
- In our own code: at most 12 model steps and USD 0.10 per agent run.

Starter code (`llm.py`, `tracing.py`, `tests/test_setup.py`) comes from the
course template `patrykbart/llm-vlm-agents-course-template`.
