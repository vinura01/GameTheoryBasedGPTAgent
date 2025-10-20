
import numpy as np

def best_responses(payoff_matrix, i, j):
    # payoff_matrix shape: (A, B, 2) -> (p1,p2)
    A, B, _ = payoff_matrix.shape
    # For given j, best responses for P1
    col = payoff_matrix[:, j, 0]
    max_p1 = np.max(col)
    p1_best = {a for a in range(A) if col[a] == max_p1}
    # For given i, best responses for P2
    row = payoff_matrix[i, :, 1]
    max_p2 = np.max(row)
    p2_best = {b for b in range(B) if row[b] == max_p2}
    return p1_best, p2_best

def pure_nash_profiles(payoff_matrix):
    A, B, _ = payoff_matrix.shape
    equilibria = []
    for i in range(A):
        for j in range(B):
            p1_best, p2_best = best_responses(payoff_matrix, i, j)
            if i in p1_best and j in p2_best:
                equilibria.append((i, j))
    return equilibria

def two_by_two_mixed_equilibrium(payoff_matrix):
    # Only for 2x2 games. Returns (p, q) where
    # p = Prob P1 plays action 0; q = Prob P2 plays action 0.
    if payoff_matrix.shape[0] != 2 or payoff_matrix.shape[1] != 2:
        return None, None
    # Payoffs
    # P1 payoffs
    a = payoff_matrix[0,0,0]
    b = payoff_matrix[0,1,0]
    c = payoff_matrix[1,0,0]
    d = payoff_matrix[1,1,0]
    # P2 payoffs
    e = payoff_matrix[0,0,1]
    f = payoff_matrix[0,1,1]
    g = payoff_matrix[1,0,1]
    h = payoff_matrix[1,1,1]
    # q makes P1 indifferent: q*a + (1-q)*b = q*c + (1-q)*d
    # => q*(a - b - c + d) = d - b
    denom1 = (a - b - c + d)
    if denom1 == 0:
        q = None
    else:
        q = (d - b) / denom1
    # p makes P2 indifferent: p*e + (1-p)*g = p*f + (1-p)*h
    # => p*(e - g - f + h) = h - g
    denom2 = (e - g - f + h)
    if denom2 == 0:
        p = None
    else:
        p = (h - g) / denom2
    if p is None or q is None or p < 0 or p > 1 or q < 0 or q > 1:
        return None, None
    return p, q
