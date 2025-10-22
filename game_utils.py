import numpy as np

# ------------------------
# 1. Prisoner's Dilemma Game
# ------------------------
def get_prisoners_dilemma_game() -> dict:
    """
    Returns the classic Prisoner's Dilemma game setup.

    The payoff matrix follows the convention:
        - Rows correspond to Player 1's actions.
        - Columns correspond to Player 2's actions.
        - Each cell contains a tuple (P1 payoff, P2 payoff).

    Payoff structure:
                   Player 2
                 C          D
        P1:C   (-1, -1)   (-3,  0)
        P1:D   ( 0, -3)   (-2, -2)

    Returns:
        dict: A dictionary containing the game name, description,
              available actions, and the payoff matrix.
    """
    return {
        "name": "Prisoner's Dilemma",
        "description": (
            "This is a simultaneous two-player game where each player chooses either to "
            "**Cooperate (C)** or **Defect (D)**. Mutual cooperation leads to moderate "
            "rewards for both, while defection can yield a higher individual payoff. "
            "All possible outcomes and their corresponding rewards are summarized "
            "in the payoff matrix below."
        ),
        "actions": ["Cooperate", "Defect"],
        "payoff_matrix": np.array([
            [[-1, -1], [-3,  0]],
            [[ 0, -3], [-2, -2]],
        ], dtype=float),
        "nash_equilibrium": (
            "Construct the payoff matrix, identify each player's best response to every action of the other player, " 
            "and apply the **Best Response** & **Dominant Strategy** principle. "   
            "(*A strategy is dominant if it yields a higher payoff regardless of what the other player chooses.*) "  
            "The pair of mutual best responses represents the **Nash Equilibrium**. "  
            "**In this game, the Nash Equilibrium is (Defect, Defect).** "
        )
    }

# ------------------------
# 2. Tic-Tac-Toe Game
# ------------------------
def get_tic_tac_toe() -> dict:
    """
    Returns the Tic-Tac-Toe game setup.

    The game is played on a 3x3 grid, and players take turns placing their marks (X or O)
    in empty cells. The first player to align three marks horizontally, vertically, or
    diagonally wins.

    Returns:
        dict: A dictionary containing the game name, description, and initial board state.
    """
    return {
        "name": "Tic-Tac-Toe",
        "description": (
            "A two player game where players alternate in marking X's and O's on a 3x3 grid.  "
            "The goal is to align three symbols in a row, column or diagonal. "
        ),
        "board": [[" " for _ in range(3)] for _ in range(3)],
        "nash_equilibrium": (
            "Represent all possible game states as a **decision tree** and **use backward induction"
            "/minimax algorithm**. Compute all terminal payoffs as win = +1, draw = 0 and lose = -1 "
            "and go backwards to find the optimal move at each stage. "
        )
    }

# ------------------------
# 3. First-Price Sealed-Bid Auction Game
# ------------------------
def get_first_price_auction_game() -> dict:
    """
    Returns the First-Price Sealed-Bid Auction game setup.

    Game summary:
        - Two players simultaneously submit sealed bids for an item.
        - The highest bidder wins and pays their bid amount.
        - The payoff is calculated as the item's value minus the bid for the winner,
          and zero for the loser.

    Payoff Matrix Convention:
        Rows   -> Player 1's bid
        Columns -> Player 2's bid
        Entries -> (P1 payoff, P2 payoff)

                     Player 2
                   Low Bid   High Bid
        P1: Low   (0, 0)     (0, V - High Bid)
        P1: High  (V - High Bid, 0)   (0, 0)
    Returns:
        dict: A dictionary containing the game metadata, available actions,
              and payoff matrix.
    """
    return {
        "name": "First-Price Sealed-Bid Auction",
        "description": (
            "In the First-Price Sealed-Bid Auction, two players simultaneously submit sealed bids "
            "for an item. The highest bidder wins and pays their bid amount. The payoff is calculated "
            "as the item's value minus the bid for the winner, and zero for the loser."
        ),
        "actions": ["Low Bid", "High Bid"],
        "payoff_matrix": np.array([
            [[0, 0], [0, "V - High Bid"]],
            [[ "V - High Bid", 0], [0, 0]],
        ], dtype=object),
        "nash_equilibrium": (
            "Modelled as a Bayesian Game, where each player has a private valuation vi and strategy = function bi(vi) which determines the bid amount."
            "Then use Expected Utility Maximization each bidder maximizes expected profit (πi) where"
            "πi=(vi−bi)P(win). Take derivative of expected utility with respect to bi, set to zero"
            "(first order condition) and solve for equilibrium bidding strategy where all players use the same function."
            "\n"
            "For example, in symmetric equilibrium with n risk-neutral bidders and values uniformly distributed between 0 and 1;"
            "b(v) = v(n−1)/n ←— Symmetric Bayesian Nash Equilibrium (Standard Case)"

            )
    }

# ------------------------
# 4. Ultimatum Bargaining Game
# ------------------------
def get_ultimatum_bargaining_game() -> dict:
    """
    Returns a simplified Ultimatum Bargaining game setup.

    Game summary:
        - Player 1 (Proposer) offers a split of some amount (e.g., $10).
        - Player 2 (Responder) decides whether to Accept or Reject.
        - If accepted, both players get the proposed shares.
        - If rejected, both get 0.

    This simplified 2x2 version models two possible offers:
        - Fair offer (50-50 split)
        - Unfair offer (80-20 split)

    Payoff Matrix Convention:
        Rows   -> Player 1's action (offer type)
        Columns -> Player 2's action (response)
        Entries -> (P1 payoff, P2 payoff)

                     Player 2
                   Accept     Reject
        Offer Fair   (5, 5)     (0, 0)
        Offer Unfair (8, 2)     (0, 0)

    Returns:
        dict: A dictionary containing the game metadata, available actions,
              and payoff matrix.
    """
    return {
        "name": "Ultimatum Bargaining Game",
        "description": (
            "In the Ultimatum Bargaining Game, Player 1 proposes how to split "
            "a sum of money between the two players. Player 2 can either "
            "**Accept** or **Reject** the offer. If Player 2 accepts, both players "
            "receive the proposed shares. If rejected, both players receive nothing. "
            "This version models a simple two-option case: a fair (50-50) and an "
            "unfair (80-20) offer."
        ),
        "roles": ["Proposer (P1)", "Responder (P2)"],
        "actions": {
            "Proposer": ["Offer Fair", "Offer Unfair"],
            "Responder": ["Accept", "Reject"]
        },
        "payoff_matrix": np.array([
            [[5, 5], [0, 0]],   
            [[8, 2], [0, 0]]    
        ], dtype=float),

        "nash_equilibrium": (
            "Use backward Induction. "
            "Analyze the last move where Player 2 decides whether to accept or reject. Accept if the offered amount is"
            "greater than zero (since any positive payoff is >0). Then, player 1 offers the smallest positive amount (ε) that player 2 accepts. "
            "nash Equilibrium is **Player 1 offers ε, Player 2 accepts**."
        )
    }