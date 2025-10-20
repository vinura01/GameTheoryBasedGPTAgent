
import numpy as np

PD_GAME = {
    "actions": ["Cooperate", "Defect"],
    # Payoffs: (P1,P2)
    #         P2:C         P2:D
    # P1:C  ( -1, -1 )   ( -3,  0 )
    # P1:D  (  0, -3 )   ( -2, -2 )
    "payoff_matrix": np.array([
        [[-1, -1], [-3,  0]],
        [[ 0, -3], [-2, -2]],
    ], dtype=float)
}
