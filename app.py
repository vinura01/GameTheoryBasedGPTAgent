
import os
import streamlit as st
import numpy as np
from agents.llm_agent import LLMAgent
from utils.nash import two_by_two_mixed_equilibrium, pure_nash_profiles

from games.prisoners_dilemma import PD_GAME
from games.ultimatum import ULTIMATUM_GAME
from games.auction_first_price import FIRST_PRICE_AUCTION
from games.tictactoe import TicTacToe

st.set_page_config(page_title="Game Theory GPT Agent", page_icon="🎮", layout="wide")

st.title("🎮 Game Theory GPT Agent")
st.caption("Streamlit + Ollama (Mistral) • Economic games • Nash helpers • Quick visuals")

# Sidebar: game selector and toggles
game_choice = st.sidebar.selectbox(
    "Pick a game",
    ["Prisoner's Dilemma", "Ultimatum Bargaining", "First‑Price Auction", "Tic‑Tac‑Toe"],
)

show_payoff = st.sidebar.toggle("Show payoff matrix", value=True)
show_nash   = st.sidebar.toggle("Compute Nash (simple)", value=False)
show_chart  = st.sidebar.toggle("Show payoff over rounds", value=True)

# Agent setup
model_name = st.sidebar.text_input("Ollama model", value="mistral")
system_prompt = st.sidebar.text_area(
    "Agent system prompt",
    value=(
        "You are a helpful game theory assistant. "
        "Given the current game, state, and history, suggest a good next move and briefly explain the rationale. "
        "If the opponent is likely to be non‑cooperative, adjust accordingly."
    ),
    height=100
)
agent = LLMAgent(model=model_name, system_prompt=system_prompt)

# Session state for rounds
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts per round
if "round_num" not in st.session_state:
    st.session_state.round_num = 0

# --- Helper UI ---
def render_payoff_matrix(matrix, players=("P1","P2"), actions_p1=None, actions_p2=None):
    st.subheader("Payoff Matrix")
    import pandas as pd
    # matrix: shape (A, B, 2) -> payoffs (p1,p2)
    A, B, _ = matrix.shape
    index = [f"{players[0]}:{actions_p1[i] if actions_p1 else i}" for i in range(A)]
    cols  = [f"{players[1]}:{actions_p2[j] if actions_p2 else j}" for j in range(B)]
    data = []
    for i in range(A):
        row = []
        for j in range(B):
            p1,p2 = matrix[i,j]
            row.append(f"({p1}, {p2})")
        data.append(row)
    df = pd.DataFrame(data, index=index, columns=cols)
    st.dataframe(df, use_container_width=True)

def render_round_chart():
    import matplotlib.pyplot as plt
    if not st.session_state.history:
        st.info("Play a round to see the chart.")
        return
    p1s = [r["payoff"][0] for r in st.session_state.history if r.get("payoff") is not None]
    p2s = [r["payoff"][1] for r in st.session_state.history if r.get("payoff") is not None]
    if not p1s or not p2s:
        st.info("No numeric payoffs recorded yet.")
        return
    fig = plt.figure()
    plt.plot(range(1, len(p1s)+1), p1s, marker="o", label="P1 payoff")
    plt.plot(range(1, len(p2s)+1), p2s, marker="o", label="P2 payoff")
    plt.xlabel("Round")
    plt.ylabel("Payoff")
    plt.legend()
    st.pyplot(fig)

def ask_agent(prompt):
    with st.spinner("Thinking with Mistral…"):
        reply = agent.chat(prompt)
    st.markdown(f"**Agent:** {reply}")
    return reply

# --- Game routing ---
if game_choice == "Prisoner's Dilemma":
    st.header("Prisoner's Dilemma")
    game = PD_GAME

    if show_payoff:
        render_payoff_matrix(game["payoff_matrix"], actions_p1=game["actions"], actions_p2=game["actions"])

    col1, col2 = st.columns(2)
    with col1:
        action_p1 = st.selectbox("Your move (P1)", game["actions"])
    with col2:
        action_p2 = st.selectbox("Opponent move (P2)", game["actions"], index=1)

    if st.button("Play Round"):
        i = game["actions"].index(action_p1)
        j = game["actions"].index(action_p2)
        payoff = tuple(game["payoff_matrix"][i,j])
        st.success(f"Payoff this round: P1={payoff[0]}, P2={payoff[1]}")
        st.session_state.history.append({"game":"PD","moves":(action_p1, action_p2), "payoff":payoff})
        st.session_state.round_num += 1

    if show_nash:
        st.subheader("Nash (basic analysis)")
        # Try pure strategies
        pure = pure_nash_profiles(game["payoff_matrix"])
        st.write("Pure‑strategy equilibria:", pure or "None")
        # Try 2x2 mixed (C/D only), returns prob of first action for each player
        p, q = two_by_two_mixed_equilibrium(game["payoff_matrix"])
        if p is not None:
            st.write(f"Mixed equilibrium (P1 plays {game['actions'][0]} with p={p:.3f}, "
                     f"P2 plays {game['actions'][0]} with q={q:.3f})")
        else:
            st.write("No interior mixed equilibrium for this payoff table.")

    user_prompt = st.text_area("Chat with the agent about strategy:", height=120,
                               placeholder="e.g., If the opponent defected last round, should I defect or cooperate now?")
    if st.button("Ask Agent"):
        context = {
            "game": "Prisoner's Dilemma",
            "actions": game["actions"],
            "payoff_matrix": game["payoff_matrix"].tolist(),
            "history": st.session_state.history[-5:],
        }
        ask_agent(f"Context: {context}\nUser: {user_prompt}")

    if show_chart:
        render_round_chart()

elif game_choice == "Ultimatum Bargaining":
    st.header("Ultimatum Bargaining")
    game = ULTIMATUM_GAME

    total = st.number_input("Total pie", min_value=1, value=10, step=1)
    offer = st.slider("P1 offer to P2", min_value=0, max_value=total, value=5)

    if st.button("Propose Offer"):
        accept_threshold = st.slider("P2 minimum acceptable offer (simulate)", 0, total, 4, key="ult_thresh")
        accepted = offer >= accept_threshold
        payoff = (total-offer, offer) if accepted else (0,0)
        st.info(f"P2 {'accepted ✅' if accepted else 'rejected ❌'}; Payoff P1={payoff[0]}, P2={payoff[1]}")
        st.session_state.history.append({"game":"Ultimatum","offer":offer,"accepted":accepted,"payoff":payoff})
        st.session_state.round_num += 1

    if show_payoff:
        st.write("Payoffs depend on accept/reject; toggle the slider to explore outcomes.")

    if show_nash:
        st.write("In the subgame‑perfect equilibrium (with perfectly rational P2), any positive offer is accepted; P1 offers the smallest positive unit. Real players deviate due to fairness norms.")

    user_prompt = st.text_area("Ask the agent about bargaining strategy:", height=120)
    if st.button("Ask Agent", key="ult_chat"):
        context = {"game":"Ultimatum", "history": st.session_state.history[-5:], "total": total, "offer": offer}
        ask_agent(f"Context: {context}\nUser: {user_prompt}")

    if show_chart:
        render_round_chart()

elif game_choice == "First‑Price Auction":
    st.header("First‑Price Sealed‑Bid Auction")
    game = FIRST_PRICE_AUCTION

    private_value = st.number_input("Your private value v", min_value=0.0, value=10.0, step=0.5)
    bids = st.text_input("Enter rival bids (comma‑sep)", value="6.5, 8.0, 9.2")
    try:
        rival_bids = [float(x.strip()) for x in bids.split(",") if x.strip()]
    except Exception:
        rival_bids = []

    if st.button("Submit Bid"):
        your_bid = st.slider("Your bid (simulate)", min_value=0.0, max_value=float(private_value), value=max(0.0, private_value*0.7))
        all_bids = rival_bids + [your_bid]
        max_bid = max(all_bids) if all_bids else 0.0
        winners = [i for i,b in enumerate(all_bids) if b == max_bid]
        you_win = (len(all_bids)-1) in winners and len(winners)==1
        payoff = (private_value - your_bid) if you_win else 0.0
        st.info(f"You {'win' if you_win else 'lose'}; payoff = {payoff:.2f}")
        st.session_state.history.append({"game":"Auction","your_bid":your_bid,"rivals":rival_bids,"win":you_win,"payoff":(payoff,0)})
        st.session_state.round_num += 1

    if show_payoff:
        st.write("Risk‑neutral symmetric equilibrium (Uniform[0,1] values) is b(v)= (n-1)/n * v. Use agent chat to adapt.")

    if show_nash:
        st.write("Closed‑form bidding functions exist under specific assumptions; for general inputs we rely on heuristics or learning.")

    user_prompt = st.text_area("Ask the agent about auction strategy:", height=120)
    if st.button("Ask Agent", key="auc_chat"):
        n = len(rival_bids) + 1
        context = {"game":"First‑Price Auction","n_bidders":n,"history":st.session_state.history[-5:],"v":private_value,"rival_bids":rival_bids}
        ask_agent(f"Context: {context}\nUser: {user_prompt}")

    if show_chart:
        render_round_chart()

else:  # Tic‑Tac‑Toe
    st.header("Tic‑Tac‑Toe")
    ttt = TicTacToe()
    st.write("Click to make a move; agent suggests a move given the board.")
    board = st.session_state.get("ttt_board") or [" "]*9

    cols = st.columns(3)
    for i in range(9):
        if cols[i//3].button(board[i] if board[i] != " " else "□", key=f"cell_{i}"):
            if board[i] == " ":
                board[i] = "X"
                st.session_state["ttt_board"] = board

    st.write("Board:", "".join(board))
    if st.button("Agent Suggests Move"):
        context = {"game":"TicTacToe","board":board}
        suggestion = ask_agent(f"Given a Tic‑Tac‑Toe board as list of 9 cells (row‑major), empty=' ', X is user, O is opponent. "
                               f"Return the best empty cell index (0-8) and a short reason. Board={board}")
        st.write("Use the suggested index to play O.")

    if st.button("Reset Board"):
        st.session_state["ttt_board"] = [" "]*9
        st.experimental_rerun()
