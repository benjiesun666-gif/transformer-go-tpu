"""
utils.py
Pure TPU utilities – no PyTorch dependency.
Provides random seed helper and self‑play statistics.
"""
import random
import numpy as np
from typing import Dict, List

def set_seed(seed: int):
    """
    Set Python and NumPy random seeds.
    Note: JAX uses its own PRNG keys; this only affects non‑JAX modules.
    """
    random.seed(seed)
    np.random.seed(seed)

class SelfPlayStats:
    """Simple self‑play statistics collector (pure Python/NumPy)."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def record_game(self, winner: int):
        """
        Record the outcome of one game.

        Args:
            winner: 1 = black wins, -1 = white wins, 0 = draw.
        """
        self.games_played += 1
        if winner == 1:  
            self.wins += 1
        elif winner == -1:  
            self.losses += 1
        else:
            self.draws += 1

    def get_stats(self) -> Dict[str, float]:
        """Return current statistics."""
        return {
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.wins / max(self.games_played, 1),
        }
