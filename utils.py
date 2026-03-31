"""
utils.py
纯净版 TPU 工具箱
100% 摆脱 PyTorch 依赖，仅保留核心辅助逻辑
"""
import random
import numpy as np
from typing import Dict, List

def set_seed(seed: int):
    """
    设置纯 Python 和 Numpy 的随机种子。
    注意：JAX 的种子是由 jax.random.PRNGKey 在运行时严格控制的，
    不需要在这里设置全局状态，这里仅用于传统模块。
    """
    random.seed(seed)
    np.random.seed(seed)

class SelfPlayStats:
    """自我对弈统计收集器 (纯 numpy/python 实现)"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def record_game(self, winner: int):
        """记录单局胜负"""
        self.games_played += 1
        if winner == 1:  # 黑胜
            self.wins += 1
        elif winner == -1:  # 白胜
            self.losses += 1
        else:
            self.draws += 1

    def get_stats(self) -> Dict[str, float]:
        """输出胜率统计字典"""
        return {
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.wins / max(self.games_played, 1),
        }