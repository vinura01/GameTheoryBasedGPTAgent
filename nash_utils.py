import numpy as np

# ------------------------
# 1. Best Responses
# ------------------------
def best_responses(payoff_matrix: np.ndarray, i: int, j: int) -> tuple[set[int], set[int]]:
    """
    Compute the best responses for both players at a given action pair (i, j).

    Args:
        payoff_matrix (np.ndarray): A 3D array of shape (A, B, 2),
            where the last dimension gives (P1 payoff, P2 payoff).
        i (int): Player 1's chosen action index.
        j (int): Player 2's chosen action index.

    Returns:
        tuple[set[int], set[int]]: Sets of best-response indices for P1 and P2.
    """
    A, B, _ = payoff_matrix.shape

    # Best responses for Player 1 given Player 2 plays j
    col = payoff_matrix[:, j, 0]
    max_p1 = np.max(col)
    p1_best = {a for a in range(A) if np.isclose(col[a], max_p1)}

    # Best responses for Player 2 given Player 1 plays i
    row = payoff_matrix[i, :, 1]
    max_p2 = np.max(row)
    p2_best = {b for b in range(B) if np.isclose(row[b], max_p2)}

    return p1_best, p2_best


# ------------------------
# 2. Pure Strategy Nash Equilibria
# ------------------------
def pure_nash_profiles(payoff_matrix: np.ndarray) -> list[tuple[int, int]]:
    """
    Find all pure-strategy Nash equilibria in a two-player game.

    Args:
        payoff_matrix (np.ndarray): A 3D array of shape (A, B, 2).

    Returns:
        list[tuple[int, int]]: List of (i, j) pairs that are pure Nash equilibria.
    """
    A, B, _ = payoff_matrix.shape
    equilibria = []

    for i in range(A):
        for j in range(B):
            p1_best, p2_best = best_responses(payoff_matrix, i, j)
            if i in p1_best and j in p2_best:
                equilibria.append((i, j))

    return equilibria


# ------------------------
# 3. Mixed Strategy Nash Equilibrium (2x2 only)
# ------------------------
def two_by_two_mixed_equilibrium(payoff_matrix: np.ndarray) -> tuple[float | None, float | None]:
    """
    Compute the mixed-strategy Nash equilibrium for a 2x2 game.

    Returns (p, q), where:
        - p: Probability Player 1 plays action 0.
        - q: Probability Player 2 plays action 0.

    Args:
        payoff_matrix (np.ndarray): A 3D array with shape (2, 2, 2).

    Returns:
        tuple[float | None, float | None]: (p, q) equilibrium probabilities,
        or (None, None) if no valid mixed equilibrium exists.
    """
    if payoff_matrix.shape != (2, 2, 2):
        raise ValueError("This function only supports 2x2 games.")

    # Payoffs for Player 1
    a, b, c, d = payoff_matrix[0, 0, 0], payoff_matrix[0, 1, 0], payoff_matrix[1, 0, 0], payoff_matrix[1, 1, 0]

    # Payoffs for Player 2
    e, f, g, h = payoff_matrix[0, 0, 1], payoff_matrix[0, 1, 1], payoff_matrix[1, 0, 1], payoff_matrix[1, 1, 1]

    # Solve for q (prob P2 plays action 0) that makes P1 indifferent
    denom1 = (a - b - c + d)
    q = None if np.isclose(denom1, 0) else (d - b) / denom1

    # Solve for p (prob P1 plays action 0) that makes P2 indifferent
    denom2 = (e - g - f + h)
    p = None if np.isclose(denom2, 0) else (h - g) / denom2

    # Check validity of probabilities
    if p is None or q is None or not (0 <= p <= 1) or not (0 <= q <= 1):
        return None, None

    return p, q
