import numpy as np
from backend.game_solvers import compute_nash_equilibria, builtin_game, vickrey_second_price, first_price_symmetric_bid, nash_bargaining_solution

def test_pd_equilibrium():
    A,B,_,_ = builtin_game("prisoners")
    eqs = list(compute_nash_equilibria(A,B))
    assert len(eqs) >= 1

def test_vickrey():
    out = vickrey_second_price([0.7, 0.9, 0.4])
    assert out["winner"] == 1 and abs(out["price"] - 0.7) < 1e-9

def test_first_price():
    assert abs(first_price_symmetric_bid(1.0, 5) - 0.8) < 1e-9

def test_nbs():
    pts = [(1,2),(1.5,1.2),(2,0.8),(1.8,1.5)]
    d = (0.5,0.5)
    res = nash_bargaining_solution(pts, d)
    assert res["solution"] is not None
