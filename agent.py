"""
Minimal REINFORCE agent for BalanceEnv.

No external ML framework required — pure Python + math.
Uses a linear softmax policy: π(a|s) = softmax(W · s + b)
"""

import math
import random
from typing import Sequence


def _dot(row: list[float], s: list[float]) -> float:
    return sum(w * x for w, x in zip(row, s))


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


class ReinforceAgent:
    """
    Linear-softmax policy trained with REINFORCE (Monte-Carlo policy gradient).

    Parameters
    ----------
    obs_dim   : dimensionality of the observation vector
    n_actions : number of discrete actions
    lr        : learning-rate
    gamma     : discount factor
    seed      : optional RNG seed
    """

    def __init__(
        self,
        obs_dim:   int   = 4,
        n_actions: int   = 2,
        lr:        float = 1e-2,
        gamma:     float = 0.99,
        seed:      int | None = None,
    ):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.lr        = lr
        self.gamma     = gamma
        self._rng      = random.Random(seed)

        # W[a] is a weight vector of length obs_dim; b[a] is a bias scalar
        scale = 0.01
        self.W: list[list[float]] = [
            [self._rng.gauss(0, scale) for _ in range(obs_dim)]
            for _ in range(n_actions)
        ]
        self.b: list[float] = [0.0] * n_actions

    # ── inference ──────────────────────────────────────────────────────────────

    def _logits(self, obs: Sequence[float]) -> list[float]:
        return [_dot(self.W[a], obs) + self.b[a] for a in range(self.n_actions)]

    def probs(self, obs: Sequence[float]) -> list[float]:
        return _softmax(self._logits(obs))

    def act(self, obs: Sequence[float]) -> int:
        """Sample an action from the current policy."""
        p = self.probs(obs)
        r = self._rng.random()
        cumulative = 0.0
        for a, prob in enumerate(p):
            cumulative += prob
            if r < cumulative:
                return a
        return self.n_actions - 1

    # ── training ───────────────────────────────────────────────────────────────

    def update(
        self,
        trajectory: list[tuple[list[float], int, float]],
    ) -> float:
        """
        Run one REINFORCE update over a full episode trajectory.

        Parameters
        ----------
        trajectory : list of (obs, action, reward) tuples

        Returns
        -------
        mean policy loss (for logging)
        """
        # 1. compute discounted returns
        G = 0.0
        returns: list[float] = []
        for _, _, r in reversed(trajectory):
            G = r + self.gamma * G
            returns.insert(0, G)

        # 2. normalise returns for stability
        mean_G = sum(returns) / len(returns)
        std_G  = math.sqrt(sum((g - mean_G) ** 2 for g in returns) / len(returns)) + 1e-8
        returns = [(g - mean_G) / std_G for g in returns]

        # 3. gradient ascent on log π(a|s) · G
        total_loss = 0.0
        for (obs, action, _), G_norm in zip(trajectory, returns):
            p = self.probs(obs)
            log_prob = math.log(p[action] + 1e-10)
            total_loss += -log_prob * G_norm

            # ∂log π / ∂W[a] = (1 - p[a]) * obs  for the chosen action
            # ∂log π / ∂W[k] = -p[k]      * obs  for k ≠ action
            for a in range(self.n_actions):
                grad = (1.0 - p[a]) if a == action else (-p[a])
                grad *= G_norm
                for j in range(self.obs_dim):
                    self.W[a][j] += self.lr * grad * obs[j]
                self.b[a] += self.lr * grad

        return total_loss / len(trajectory)
