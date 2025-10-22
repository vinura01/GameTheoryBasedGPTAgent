import numpy as np
import streamlit as st
from ollama_utils import is_ollama_running, get_llms, chat_with_model
from game_utils import get_prisoners_dilemma_game, get_tic_tac_toe, get_ultimatum_bargaining_game
from nash_utils import two_by_two_mixed_equilibrium, pure_nash_profiles

# ------------------------
# Streamlit page configuration
# ------------------------
def configure_page() -> None:
    st.set_page_config(
        page_title="Game Theory GPT Agent",
        page_icon="🎮",
        layout="wide"
    )

    st.title("🎮 Game Theory GPT Agent")
    st.caption("Streamlit + Ollama • Economic games • Nash helpers • Quick visuals")

# ------------------------
# Sidebar: select model
# ------------------------
def configure_sidebar() -> tuple[str, bool, bool, bool, str | None, str]:
    """Returns (game_choice, show_payoff, show_nash, show_chart, selected_model, system_prompt)"""
    selected_model = None

    with st.sidebar:
        game_choice = st.selectbox(
            "Pick a game",
            ["Prisoner's Dilemma", "Tic-Tac-Toe", "First-Price Auction", "Ultimatum Bargaining"],
            key="sidebar_game_choice"
        )

        show_payoff = st.checkbox("Show payoff matrix", value=True, key="sidebar_show_payoff")
        show_nash = st.checkbox("Theoretical Equilibrium", value=False, key="sidebar_show_nash")
        show_chart = st.checkbox("Show payoff over rounds", value=True, key="sidebar_show_chart")

        st.divider()

        if is_ollama_running():
            st.success("✅ Ollama is running")
            models = get_llms()
            if models:
                selected_model = st.selectbox("Choose a model", models, key="model_select")
            else:
                st.warning("⚠️ No models available. Use `ollama pull <model>` to add one.")
        else:
            st.error("❌ Ollama is not running. Please start it first.")

        system_prompt = st.text_area(
            "Agent system prompt",
            value=(
                "You are a strategic game-theory assistant. "
                "Analyze the current game, including its state and past moves, and recommend the best next action. "
                "Briefly explain your reasoning, considering both optimal play and possible non-cooperative behavior from the opponent."
            ),
            height=150,
            key="sidebar_system_prompt"
        )

    return game_choice, show_payoff, show_nash, show_chart, selected_model, system_prompt

# ------------------------
# initialize history
# ------------------------
def initialize_history() -> None:
    # Session state for history
    if "history" not in st.session_state:
        st.session_state.history = [] 
    # Session state for rounds
    if "round_num" not in st.session_state:
        st.session_state.round_num = 0
    # TicTacToe board state
    if "ttt_board" not in st.session_state:
        st.session_state["ttt_board"] = [" "] * 9

# ------------------------
# Payoff matrix rendering
# ------------------------
def render_payoff_matrix(matrix, players=("P1", "P2"), actions_p1=None, actions_p2=None) -> None:
    st.subheader("Payoff Matrix")
    import pandas as pd
    matrix = np.array(matrix)
    if matrix.ndim != 3 or matrix.shape[2] != 2:
        st.error("Payoff matrix must have shape (A, B, 2).")
        return
    A, B, _ = matrix.shape
    index = [f"{players[0]}:{(actions_p1[i] if actions_p1 else i)}" for i in range(A)]
    cols = [f"{players[1]}:{(actions_p2[j] if actions_p2 else j)}" for j in range(B)]
    data = []
    for i in range(A):
        row = []
        for j in range(B):
            p1, p2 = matrix[i, j]
            row.append(f"({p1}, {p2})")
        data.append(row)
    df = pd.DataFrame(data, index=index, columns=cols)
    st.dataframe(df, use_container_width=True)

# ------------------------
# Round chart rendering
# ------------------------
def render_round_chart() -> None:
    import matplotlib.pyplot as plt
    if not st.session_state.get("history"):
        st.info("Play a round to see the chart.")
        return
    p1s = [r["payoff"][0] for r in st.session_state.history if r.get("payoff") is not None]
    p2s = [r["payoff"][1] for r in st.session_state.history if r.get("payoff") is not None]
    if not p1s or not p2s:
        st.info("No numeric payoffs recorded yet.")
        return
    fig = plt.figure()
    plt.plot(range(1, len(p1s) + 1), p1s, marker="o", label="P1 payoff")
    plt.plot(range(1, len(p2s) + 1), p2s, marker="o", label="P2 payoff")
    plt.xlabel("Round")
    plt.ylabel("Payoff")
    plt.legend()
    st.pyplot(fig)

#------------------------
# Prisoners dilemma
#------------------------
def prisoners_dilemma(show_payoff: bool, show_chart: bool) -> tuple[dict, dict, str]:
    game = get_prisoners_dilemma_game()

    st.header(game["name"])
    st.markdown(game["description"])

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            if show_payoff:
                render_payoff_matrix(game["payoff_matrix"], actions_p1=game["actions"], actions_p2=game["actions"])
            
            col3, col4 = st.columns(2)
            with col3:
                action_p1 = st.selectbox("Your move (P1)", game["actions"], key="pd_p1")
            with col4:
                default_idx = 1 if len(game["actions"]) > 1 else 0
                action_p2 = st.selectbox("Opponent move (P2)", game["actions"], index=default_idx, key="pd_p2")

            if st.button("Play Round", key="pd_play"):
                i = game["actions"].index(action_p1)
                j = game["actions"].index(action_p2)
                payoff = tuple(game["payoff_matrix"][i, j])
                st.success(f"Payoff this round: P1={payoff[0]}, P2={payoff[1]}")
                st.session_state.history.append({"game": "PD", "moves": (action_p1, action_p2), "payoff": payoff})
                st.session_state.round_num += 1

        with col2:
            if show_chart:
                render_round_chart()
        
    full_prompt = {
                "game": "Prisoner's Dilemma",
                "actions": game["actions"],
                "payoff_matrix": np.array(game["payoff_matrix"]).tolist(),
                "history": st.session_state.history[-5:],
            }
    
    placeholder_text="e.g., If the opponent defected last round, should I defect or cooperate now?"

    return game, full_prompt, placeholder_text

#------------------------
# Tic-Tac-Toe
#------------------------
def tic_tac_toe(show_payoff: bool, show_chart: bool) -> tuple[dict, dict, str]:
    game = get_tic_tac_toe()
    st.header(game["name"])
    st.write(game["description"])

    # ------------------------
    # Initialize state
    # ------------------------
    if "ttt_board" not in st.session_state:
        st.session_state["ttt_board"] = [" "] * 9
    if "ttt_turn" not in st.session_state:
        st.session_state["ttt_turn"] = "X"
    if "ttt_winner" not in st.session_state:
        st.session_state["ttt_winner"] = None

    board = st.session_state["ttt_board"]
    turn = st.session_state["ttt_turn"]
    winner = st.session_state["ttt_winner"]

    # ------------------------
    # Helper: check winner
    # ------------------------
    def check_winner(b):
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for combo in win_combinations:
            if b[combo[0]] == b[combo[1]] == b[combo[2]] != " ":
                return b[combo[0]]
        return None

    # ------------------------
    # Show current turn or winner
    # ------------------------
    if winner:
        st.markdown(f"### :red[Player {winner} Wins! 🎉]")
    else:
        st.subheader(f"Current Turn: {turn}")

    # ------------------------
    # Render 3x3 grid
    # ------------------------
    cols = st.columns(3)
    for i in range(9):
        with cols[i % 3]:
            disabled = board[i] != " " or winner is not None
            if st.button(board[i] if board[i] != " " else "□", key=f"cell_{i}", disabled=disabled):
                if board[i] == " " and winner is None:
                    board[i] = turn
                    st.session_state["ttt_turn"] = "O" if turn == "X" else "X"
                    st.session_state["ttt_board"] = board
                    win = check_winner(board)
                    if win:
                        st.session_state["ttt_winner"] = win
                    st.rerun()

    # ------------------------
    # Reset
    # ------------------------
    if st.button("Reset Board"):
        st.session_state["ttt_board"] = [" "] * 9
        st.session_state["ttt_turn"] = "X"
        st.session_state["ttt_winner"] = None
        st.rerun()

    # ------------------------
    # Prepare prompt for chatbot
    # ------------------------
    full_prompt = {
        "game": "Tic-Tac-Toe",
        "board": board,
        "current_turn": turn,
        "winner": winner,
        "prompt": (
            f"The current Tic-Tac-Toe board is {board}. "
            f"It is player {turn}'s turn. "
            f"Suggest the optimal next move(s) or strategy. "
            f"Use 0–8 indexing for board positions (left-to-right, top-to-bottom). "
            f"Explain why the suggested move is best. "
            f"If the game is over, simply summarize the outcome."
        )
    }

    placeholder_text = (
        "Ask about your next best move — e.g., 'What’s the optimal next move for X?'"
    )

    # ------------------------
    # Display current board
    # ------------------------
    st.write("Board:", "".join(board))

    return game, full_prompt, placeholder_text




# ------------------------
# First-Price Auction Game
# ------------------------
def first_price_auction_game():
    ...

# ------------------------
# Ultimatum Bargaining Game
# ------------------------
def ultimatum_bargaining_game(show_payoff: bool, show_chart: bool) -> tuple[dict, dict, str]:
    game = get_ultimatum_bargaining_game()

    st.header(game["name"])
    st.write(game["description"])

    total = 100
    col1, col2 = st.columns(2)
    with col1:
        offer = st.slider("Player 1 Offer (0–100)", 0, total, 50, step=5, key="ug_offer")
    with col2:
        response = st.selectbox("Player 2 Decision", ["Accept", "Reject"], key="ug_response")

    # ------------------------
    # Compute payoffs
    # ------------------------
    if st.button("Play Round", key="ug_play"):
        if response == "Accept":
            payoff = (total - offer, offer)
            st.success(f"✅ Offer accepted! P1 gets {payoff[0]}, P2 gets {payoff[1]}")
        else:
            payoff = (0, 0)
            st.error("❌ Offer rejected! Both players get 0.")

        st.session_state.history.append({
            "game": "Ultimatum",
            "offer": offer,
            "response": response,
            "payoff": payoff
        })
        st.session_state.round_num += 1

    # ------------------------
    # Optional payoff table and chart side by side
    # ------------------------
    if show_payoff or show_chart:
        st.subheader("Visual Insights")

        col_table, col_chart = st.columns([1, 1.2])

        with col_table:
            if show_payoff:
                import pandas as pd
                st.markdown("**Payoffs if Accepted**")
                data = [{"Offer": o, "P1 Payoff": total - o, "P2 Payoff": o} for o in range(0, total + 1, 20)]
                df = pd.DataFrame(data)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )

        with col_chart:
            if show_chart:
                st.markdown("**Offer History Chart**")
                render_round_chart()

    # ------------------------
    # Prompt for AI agent
    # ------------------------
    full_prompt = {
        "game": "Ultimatum Bargaining Game",
        "total": total,
        "last_offer": offer,
        "last_response": response,
        "history": st.session_state.history[-5:],
        "prompt": (
            f"In an Ultimatum Bargaining Game with 100 units, Player 1 offered {offer} units to Player 2. "
            f"Player 2 chose to '{response}'. "
            f"Discuss whether this was rational from both sides' perspectives. "
            f"Suggest an optimal offer strategy for Player 1 and an acceptance threshold for Player 2."
        ),
    }

    placeholder_text = (
        "Ask the agent: 'What’s a fair offer that’s likely to be accepted?' or "
        "'Why might rejecting low offers be irrational?'"
    )

    return game, full_prompt, placeholder_text



# ------------------------
# Theoretical Nash Equilibrium Display
# ------------------------   
# ------------------------
# Theoretical Nash Equilibrium Display
# ------------------------
def theoretical_nash_equilibrium(game: dict, show_nash: bool) -> None:
    if show_nash:
        with st.container(border=True):
            st.subheader("Theoretical Nash Equilibrium")
            st.markdown(game["nash_equilibrium"])

            # Show pure strategy equilibria for all games except Ultimatum Bargaining
            if game.get("name") != "Ultimatum Bargaining Game":
                pure = pure_nash_profiles(np.array(game["payoff_matrix"]))
                st.write("Pure-strategy equilibria:", pure or "None")

            # Always show mixed equilibrium (if 2x2)
            try:
                p, q = two_by_two_mixed_equilibrium(np.array(game["payoff_matrix"]))
                if p is not None:
                    st.write(
                        f"Mixed equilibrium (P1 plays {game['actions'][0]} with p={p:.3f}, "
                        f"P2 plays {game['actions'][0]} with q={q:.3f})"
                    )
                else:
                    st.write("No interior mixed equilibrium for this payoff table.")
            except Exception:
                st.write("Mixed equilibrium not applicable for this game.")

    
# ------------------------
# GPT Agent Interaction
# ------------------------
def query_gpt_agent(model: str , prompt: str) -> None:
    with st.spinner("Thinking..."):
        with st.expander("Response", expanded=True):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in chat_with_model(model, prompt):
                if any(tag in chunk for tag in ["<think>", "</think>"]):
                    continue
                full_response += chunk
                response_placeholder.markdown(f"{full_response}")

# ------------------------
# Agent Chat Interface
# ------------------------
def render_ai_agent_ui(model: str, full_prompt: str, placeholder_text: str, system_prompt: str) -> None:
    with st.container(border=True):
        user_prompt = st.text_area("Chat with the agent about strategy:", height=120,
                                placeholder=placeholder_text,
                                key="pd_user_prompt")
        
        if st.button("Ask Agent", key="pd_ask_agent"):
            query_gpt_agent(model, f"Context: {full_prompt} \n System: {system_prompt} \n User: {user_prompt}")

# ------------------------
# Main app logic
# ------------------------
def main():
    configure_page()

    game_choice, show_payoff, show_nash, show_chart, selected_model, system_prompt = configure_sidebar()

    if not selected_model:
        st.info("Select a model from the sidebar to begin.")
        return

    initialize_history()

    # ------------------------
    # Route to chosen game
    # ------------------------
    if game_choice == "Prisoner's Dilemma":
        game, full_prompt, placeholder_text = prisoners_dilemma(show_payoff, show_chart)
    elif game_choice == "Tic-Tac-Toe":
        game, full_prompt, placeholder_text = tic_tac_toe(show_payoff, show_chart)
    elif game_choice == "Ultimatum Bargaining":
        game, full_prompt, placeholder_text = ultimatum_bargaining_game(show_payoff, show_chart)
    else:
        st.warning("Selected game not implemented yet.")
        return

    # ------------------------
    # Theoretical Nash Equilibrium (safe rendering)
    # ------------------------
    try:
        if show_nash:
            theoretical_nash_equilibrium(game, show_nash)
    except Exception as e:
        st.warning(f"⚠️ Unable to compute Nash Equilibrium: {e}")

    # ------------------------
    # Always render the AI agent last in its own container
    # ------------------------
    with st.container():
        st.divider()
        render_ai_agent_ui(selected_model, full_prompt, placeholder_text, system_prompt)


if __name__ == "__main__":
    main()
