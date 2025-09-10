# Algorithmic Game Theory GPT Agent

A local, privacy-friendly agent for solving classic Algorithmic Game Theory problems with a clean Streamlit dashboard and an optional FastAPI backend.
It uses **nashpy** for 2-player normal-form Nash equilibria, plus small calculators for auctions and bargaining.
Optional LLM routing via **Ollama** (Mistral 7B) can parse free-text problems into structured inputs.

## Features

- Two-player normal-form games: pure & mixed Nash equilibria (via `nashpy`), best responses, payoffs.
- Built-ins: Prisoner’s Dilemma, Matching Pennies, Battle of the Sexes, Coordination, Rock–Paper–Scissors.
- Auctions: Vickrey (2nd-price) outcome; First-price (symmetric, risk-neutral, private values ~ U[0,1]) equilibrium bid function `b(v) = (n-1)/n * v` (shows expected revenue).
- Bargaining: Nash bargaining solution for 2 players given utility frontier (discrete points) and a disagreement point `d`.
- Tic-tac-toe: simple minimax agent for best move.
- LLM Router (optional): use Ollama (e.g., `mistral:7b`) to parse natural-language prompts into a structured schema that the calculators can solve.
- Streamlit dashboard with editable payoff matrices and plots.
- FastAPI backend with REST endpoints for programmatic calls (and for your own UI stacks).

## Quickstart

```bash
# 1) Create & activate a virtualenv (recommended Python 3.10+)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 2) Install requirements
pip install -r requirements.txt

# 3) (Optional) Start FastAPI backend on :8010
uvicorn backend.app:app --reload --port 8010

# 4) Start Streamlit dashboard on :8501
streamlit run ui/streamlit_app.py
```

### Optional: Ollama LLM

```bash
# Install Ollama and pull a model (example)
ollama pull mistral:7b
# (or use any model you prefer and set in .env)
```

Create a `.env` file (or set environment variables) for the LLM:

```
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=mistral:7b
BACKEND_URL=http://127.0.0.1:8010
```

## Project Layout

```
backend/
  app.py               # FastAPI endpoints
  game_solvers.py      # Core solvers (Nash, auctions, bargaining, tictactoe)
  llm_router.py        # Optional: NL→structured routing via Ollama
ui/
  streamlit_app.py     # Dashboard with matrix editor, plots, and API calls
tests/
  test_game_solvers.py # Minimal sanity tests
README.md
requirements.txt
.env
```

## Notes

- Mixed-strategy NE can be numerically delicate; `nashpy` handles common cases well.
- First-price auction function assumes symmetric independent private values ~ U[0,1] and risk-neutral bidders.
- Bargaining solver approximates the Pareto frontier from a discrete set and maximizes the Nash product `(u1-d1)*(u2-d2)`.
- Tic-tac-toe is small enough for minimax; depth-limited search keeps UI responsive.
