from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
import itertools
import nashpy as nash

# ---------- Normal-form games ----------

def compute_nash_equilibria(A: np.ndarray, B: np.ndarray):
    """
    Return list of (sigma_r, sigma_c) mixed-strategy profiles that are Nash equilibria.
    A, B are payoff matrices for Row and Column players respectively (same shape).
    """
    game = nash.Game(A, B)
    equilibria = list(game.support_enumeration())  # generator -> list
    return equilibria

def best_responses(A: np.ndarray, B: np.ndarray):
    """
    Given payoff matrices A (row player) and B (column player), compute best responses.
    Returns dict with pure best responses for each opponent pure action.
    """
    br_row = {}
    br_col = {}
    # Against each column pure j, best rows maximize A[:, j]
    for j in range(B.shape[1]):
        col_payoffs = A[:, j]
        br_row[j] = np.where(col_payoffs == col_payoffs.max())[0].tolist()
    # Against each row pure i, best columns maximize B[i, :]
    for i in range(A.shape[0]):
        row_payoffs = B[i, :]
        br_col[i] = np.where(row_payoffs == row_payoffs.max())[0].tolist()
    return {"row_best_responses": br_row, "col_best_responses": br_col}

def builtin_game(name: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    name = name.lower().strip()
    if name in ["prisoners", "prisoner's dilemma", "prisoners dilemma", "pd"]:
        # Actions: C, D
        # Payoffs (Row, Col)
        A = np.array([[ -1, -10],
                      [  0,  -5]])
        B = np.array([[ -1,   0],
                      [-10,  -5]])
        return A, B, ["C","D"], ["C","D"]
    if name in ["matching pennies", "mp"]:
        A = np.array([[ 1, -1],
                      [-1,  1]])
        B = -A
        return A, B, ["H","T"], ["H","T"]
    if name in ["battle of the sexes", "bos"]:
        A = np.array([[2,0],
                      [0,1]])
        B = np.array([[1,0],
                      [0,2]])
        return A, B, ["Ballet","Football"], ["Ballet","Football"]
    if name in ["coordination"]:
        A = np.array([[1,0],
                      [0,1]])
        B = A.copy()
        return A, B, ["L","R"], ["L","R"]
    if name in ["rock paper scissors","rps"]:
        A = np.array([[0,-1, 1],
                      [1, 0,-1],
                      [-1,1, 0]])
        B = -A
        return A, B, ["R","P","S"], ["R","P","S"]
    raise ValueError(f"Unknown builtin game: {name}")

# ---------- Auctions ----------

def vickrey_second_price(valuations: List[float]) -> Dict:
    """
    Computes winner and payment in a single-item sealed-bid second-price auction (truthful).
    valuations: list of private values (assumed equal to bids in DSIC mechanism).
    """
    n = len(valuations)
    order = np.argsort(valuations)[::-1]  # descending
    winner = int(order[0])
    price = float(valuations[order[1]]) if n > 1 else 0.0
    return {"winner": winner, "price": price, "valuations": valuations}

def first_price_symmetric_bid(v: float, n_bidders: int) -> float:
    """Risk-neutral IPV ~ U[0,1] symmetric equilibrium bid: b(v) = (n-1)/n * v"""
    if not (0 <= v <= 1): 
        raise ValueError("Assumes v in [0,1]")
    if n_bidders < 2:
        raise ValueError("n_bidders must be >= 2")
    return (n_bidders - 1) / n_bidders * v

# ---------- Bargaining ----------

def nash_bargaining_solution(points: List[Tuple[float,float]], d: Tuple[float,float]) -> Dict:
    """
    Approximate NBS over discrete feasible utility set `points` with disagreement point d.
    Maximizes (u1-d1)*(u2-d2) subject to u1>=d1, u2>=d2.
    """
    d1, d2 = d
    feasible = [(u1,u2) for (u1,u2) in points if u1>=d1 and u2>=d2]
    if not feasible:
        return {"solution": None, "objective": -float("inf")}
    best = None
    best_val = -1
    for (u1,u2) in feasible:
        val = (u1-d1)*(u2-d2)
        if val > best_val:
            best_val = val
            best = (u1,u2)
    return {"solution": best, "objective": best_val}

# ---------- Tic-tac-toe ----------

def ttt_best_move(board: List[str], player: str) -> Dict:
    """
    board: list of 9 strings in {'X','O',' '}.
    player: 'X' or 'O' to move.
    Returns best move index and resulting score via minimax.
    """
    def winner(b):
        lines = [(0,1,2),(3,4,5),(6,7,8),
                 (0,3,6),(1,4,7),(2,5,8),
                 (0,4,8),(2,4,6)]
        for i,j,k in lines:
            if b[i] != ' ' and b[i]==b[j]==b[k]:
                return b[i]
        if ' ' not in b:
            return 'D'  # draw
        return None

    def minimax(b, p, maximizing):
        w = winner(b)
        if w == 'X': return 1
        if w == 'O': return -1
        if w == 'D': return 0
        scores = []
        for idx in range(9):
            if b[idx] == ' ':
                b[idx] = p
                score = minimax(b, 'O' if p=='X' else 'X', not maximizing)
                b[idx] = ' '
                scores.append(score)
        return max(scores) if maximizing else min(scores)

    best_idx = None
    best_score = -10 if player=='X' else 10
    for idx in range(9):
        if board[idx] == ' ':
            board[idx] = player
            score = minimax(board, 'O' if player=='X' else 'X', player=='O')
            board[idx] = ' '
            if player=='X':
                if score > best_score:
                    best_score, best_idx = score, idx
            else:
                if score < best_score:
                    best_score, best_idx = score, idx
    return {"move": best_idx, "score": best_score}
