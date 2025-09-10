import os
import json
import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8010")

st.set_page_config(page_title="Algorithmic Game Theory Agent", layout="wide")
st.title("🎯 Algorithmic Game Theory Agent")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Normal-form (Nash)",
    "Built-in Games",
    "Auctions",
    "Bargaining",
    "Tic-Tac-Toe"
])

with tab1:
    st.subheader("Two-player normal-form game")
    r = st.number_input("Rows (Row player actions)", 2, 10, 2)
    c = st.number_input("Cols (Column player actions)", 2, 10, 2)

    st.write("Enter payoffs for **Row player (A)**:")
    A = np.zeros((r,c))
    for i in range(r):
        cols = st.columns(c)
        for j in range(c):
            A[i,j] = cols[j].number_input(f"A[{i},{j}]", value=0.0, key=f"A_{i}_{j}")

    st.write("Enter payoffs for **Column player (B)**:")
    B = np.zeros((r,c))
    for i in range(r):
        cols = st.columns(c)
        for j in range(c):
            B[i,j] = cols[j].number_input(f"B[{i},{j}]", value=0.0, key=f"B_{i}_{j}")

    if st.button("Compute Nash equilibria"):
        try:
            resp = requests.post(f"{BACKEND}/equilibria", json={"A": A.tolist(), "B": B.tolist()})
            data = resp.json()
            st.success("Done.")
            st.json(data)
        except Exception as e:
            st.error(str(e))

    st.write("Heatmap of Row payoffs (A)")
    fig = plt.figure()
    plt.imshow(A)
    plt.colorbar()
    st.pyplot(fig)

with tab2:
    st.subheader("Built-in Games")
    name = st.selectbox("Choose", ["Prisoner's Dilemma", "Matching Pennies", "Battle of the Sexes", "Coordination", "Rock Paper Scissors"])
    if st.button("Load"):
        resp = requests.post(f"{BACKEND}/builtin", json={"name": name})
        game = resp.json()
        st.json(game)
        if st.button("Compute Equilibria for Built-in"):
            resp2 = requests.post(f"{BACKEND}/equilibria", json={"A": game["A"], "B": game["B"]})
            st.json(resp2.json())

with tab3:
    st.subheader("Auctions")
    atype = st.selectbox("Auction type", ["Vickrey (Second-price)", "First-price (Symmetric, U[0,1])"])
    if atype.startswith("Vickrey"):
        vals = st.text_input("Valuations as comma-separated floats", "0.7, 0.4, 0.9, 0.5")
        if st.button("Compute Vickrey outcome"):
            valuations = [float(x.strip()) for x in vals.split(",") if x.strip()]
            resp = requests.post(f"{BACKEND}/auction/vickrey", json={"valuations": valuations})
            st.json(resp.json())
    else:
        v = st.slider("Your valuation v (0..1)", 0.0, 1.0, 0.8, 0.01)
        n = st.number_input("Number of bidders", 2, 20, 4)
        if st.button("Equilibrium Bid"):
            resp = requests.post(f"{BACKEND}/auction/first_price", json={"v": v, "n_bidders": int(n)})
            st.json(resp.json())

with tab4:
    st.subheader("Bargaining (Nash)")
    st.markdown("Enter feasible utility points and a disagreement point.")
    points_str = st.text_area("Points as JSON (e.g., [[1,2],[1.5,1.2],[2,0.8]])", "[[1,2],[1.5,1.2],[2,0.8],[1.8,1.5]]")
    d_str = st.text_input("Disagreement point d as JSON", "[0.5,0.5]")
    if st.button("Solve NBS"):
        try:
            points = json.loads(points_str)
            d = json.loads(d_str)
            resp = requests.post(f"{BACKEND}/bargain/nbs", json={"points": points, "d": d})
            res = resp.json()
            st.json(res)
        except Exception as e:
            st.error(str(e))

with tab5:
    st.subheader("Tic-Tac-Toe")
    st.write("Use 'X', 'O', or space ' ' for empty. Indexing is 0..8.")
    default_board = [" "]*9
    board = []
    cols = st.columns(3)
    for i in range(9):
        if i % 3 == 0 and i>0:
            cols = st.columns(3)
        board.append(cols[i%3].text_input(f"{i}", " ", max_chars=1, key=f"b{i}"))
    player = st.selectbox("Player to move", ["X","O"])
    if st.button("Best move"):
        resp = requests.post(f"{BACKEND}/tictactoe/best", json={"board": board, "player": player})
        st.json(resp.json())
