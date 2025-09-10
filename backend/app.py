from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple, Dict, Optional
import numpy as np
from .game_solvers import compute_nash_equilibria, best_responses, builtin_game, vickrey_second_price, first_price_symmetric_bid, nash_bargaining_solution, ttt_best_move

app = FastAPI(title="Game Theory Agent API")

class MatrixGame(BaseModel):
    A: List[List[float]]
    B: List[List[float]]

class Builtin(BaseModel):
    name: str

class Vickrey(BaseModel):
    valuations: List[float]

class FirstPrice(BaseModel):
    v: float
    n_bidders: int

class Bargain(BaseModel):
    points: List[Tuple[float,float]]
    d: Tuple[float,float]

class TTT(BaseModel):
    board: List[str]
    player: str

@app.post("/equilibria")
def equilibria(game: MatrixGame):
    A = np.array(game.A, dtype=float)
    B = np.array(game.B, dtype=float)
    eqs = compute_nash_equilibria(A, B)
    readable = []
    for s_row, s_col in eqs:
        readable.append({
            "row_strategy": s_row.tolist(),
            "col_strategy": s_col.tolist(),
            "row_sum": float(np.sum(s_row)),
            "col_sum": float(np.sum(s_col)),
        })
    return {"equilibria": readable, "best_responses": best_responses(A, B)}

@app.post("/builtin")
def builtin(b: Builtin):
    A,B,rows,cols = builtin_game(b.name)
    return {"A": A.tolist(), "B": B.tolist(), "rows": rows, "cols": cols}

@app.post("/auction/vickrey")
def auction_vickrey(v: Vickrey):
    return vickrey_second_price(v.valuations)

@app.post("/auction/first_price")
def auction_first_price(p: FirstPrice):
    bid = first_price_symmetric_bid(p.v, p.n_bidders)
    return {"bid": bid}

@app.post("/bargain/nbs")
def bargain_nbs(b: Bargain):
    res = nash_bargaining_solution(b.points, b.d)
    return res

@app.post("/tictactoe/best")
def tictactoe_best(ttt: TTT):
    return ttt_best_move(ttt.board, ttt.player)
