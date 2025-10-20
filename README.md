# Game Theory GPT Agent (Streamlit + Ollama)

A lightweight playground for economic/game‑theory experiments with a local LLM (Ollama + Mistral).
Pick a game on the left, toggle visualizations (payoff matrix, graphs), and chat with an agent that suggests moves/strategies.

## Features
- **Games**: Prisoner's Dilemma, Ultimatum Bargaining, First‑Price Sealed‑Bid Auction, Tic‑Tac‑Toe.
- **Local LLM Agent**: Uses Ollama (`mistral`) via HTTP chat API (no cloud calls).
- **Nash Utilities**: Quick helpers for 2×2 mixed strategies + pure‑strategy scan.
- **Toggles**: Show payoff matrix, compute Nash (simple), show round chart.
- **LAN Access**: Streamlit config listens on `0.0.0.0` so other devices on your Wi‑Fi can open it.

## Prereqs
- Python 3.10+ recommended
- [Ollama](https://ollama.com/) installed and running: `ollama serve`
- Pull a model (Mistral 7B): `ollama pull mistral` (or another chat‑capable model)

## Quick Start
```bash
cd game_theory_agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# ensure ollama is running locally
# mac/linux:
ollama serve &

# Run Streamlit (served on 0.0.0.0:8501 so LAN devices can access)
streamlit run app.py
```
Then from your other device on the same network, open:
```
http://<YOUR_LAN_IP>:8501
```
Example: `http://192.168.1.215:8501`

> If port is busy, change it in `.streamlit/config.toml` or run:
> `streamlit run app.py --server.port 8502`

## Dev Notes
- The agent calls Ollama at `http://localhost:11434/api/chat`. If Ollama runs elsewhere,
  set `OLLAMA_HOST=http://<host>:11434` in your environment.
- This project keeps math simple and readable. The included Nash methods cover basic 2×2 games
  and a simple pure‑strategy scan for small payoff tables. Extend as you like!
- All state is in-memory; no database required.
