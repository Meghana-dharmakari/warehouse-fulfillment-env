"""
tasks.py — 3 graded fulfillment tasks (easy → medium → hard).

Each task returns a TaskResult with score ∈ [0.0, 1.0].
"""

from __future__ import annotations
from dataclasses import dataclass
from env import WarehouseFulfillmentEnv
from models import WarehouseState


@dataclass
class TaskResult:
    task_name: str
    score: float          # 0.0 – 1.0
    details: dict


# ── grader helpers ─────────────────────────────────────────────────────────────

def _fill_rate_score(state: WarehouseState) -> float:
    """Mean fill-rate across all orders."""
    orders = state.orders + state.shipped_orders
    if not orders:
        return 0.0
    return sum(o.total_fill_rate for o in orders) / len(orders)


def _on_time_ship_rate(state: WarehouseState) -> float:
    """Fraction of shipped orders that were on time."""
    shipped = [o for o in state.shipped_orders]
    if not shipped:
        return 0.0
    on_time = sum(1 for o in shipped if not o.late)
    return on_time / len(shipped)


def _throughput_score(state: WarehouseState, target: int) -> float:
    """Fraction of target orders shipped."""
    return min(1.0, len(state.shipped_orders) / target)


# ── Task 1 — Easy: ship any 2 orders fully ────────────────────────────────────

def run_task_easy(agent_fn, seed: int = 0) -> TaskResult:
    """
    Task: Ship at least 2 fully-fulfilled orders within 50 steps.
    Score = fraction of 2 target orders shipped fully (0.0 – 1.0).
    """
    env = WarehouseFulfillmentEnv(n_orders=3, max_steps=50, seed=seed)
    obs = env.reset()
    done = False
    while not done:
        action = agent_fn(obs)
        result = env.step(action)
        obs, done = result.observation, result.done

    state = env.state()
    fully_shipped = sum(
        1 for o in state.shipped_orders if o.fully_fulfilled and not o.late
    )
    score = min(1.0, fully_shipped / 2)
    return TaskResult(
        task_name="easy_ship_2_orders",
        score=round(score, 4),
        details={
            "fully_shipped_on_time": fully_shipped,
            "target": 2,
            "total_reward": state.total_reward,
        },
    )


# ── Task 2 — Medium: ≥80 % fill-rate + ≥50 % on-time ─────────────────────────

def run_task_medium(agent_fn, seed: int = 0) -> TaskResult:
    """
    Task: Achieve ≥80 % mean fill-rate AND ≥50 % on-time ship rate
          across 5 orders within 60 steps.
    Score = harmonic mean of (fill_rate / 0.8) and (on_time_rate / 0.5),
            capped at 1.0.
    """
    env = WarehouseFulfillmentEnv(n_orders=5, max_steps=60, seed=seed)
    obs = env.reset()
    done = False
    while not done:
        action = agent_fn(obs)
        result = env.step(action)
        obs, done = result.observation, result.done

    state = env.state()
    fr  = _fill_rate_score(state)
    otr = _on_time_ship_rate(state)

    fr_norm  = min(1.0, fr  / 0.80)
    otr_norm = min(1.0, otr / 0.50)

    if fr_norm + otr_norm == 0:
        score = 0.0
    else:
        score = 2 * fr_norm * otr_norm / (fr_norm + otr_norm)   # harmonic mean

    return TaskResult(
        task_name="medium_fill_and_ontime",
        score=round(score, 4),
        details={
            "fill_rate":      round(fr, 4),
            "on_time_rate":   round(otr, 4),
            "shipped":        len(state.shipped_orders),
            "total_reward":   state.total_reward,
        },
    )


# ── Task 3 — Hard: high throughput + efficiency under scarcity ────────────────

def run_task_hard(agent_fn, seed: int = 0) -> TaskResult:
    """
    Task: Ship ≥7 of 10 orders fully and on-time within 80 steps,
          with tight inventory (low initial stock).
    Score = weighted combination:
              0.5 * throughput_score  +  0.3 * fill_rate  +  0.2 * on_time_rate
    """
    env = WarehouseFulfillmentEnv(
        skus=["SKU-A", "SKU-B", "SKU-C", "SKU-D", "SKU-E"],
        n_orders=10,
        max_steps=80,
        seed=seed,
    )
    # Tighten inventory after reset
    obs = env.reset()
    for sku in env.skus:
        env._state.inventory[sku] = max(3, env._state.inventory[sku] // 3)

    done = False
    while not done:
        action = agent_fn(obs)
        result = env.step(action)
        obs, done = result.observation, result.done

    state = env.state()
    tp  = _throughput_score(state, target=7)
    fr  = _fill_rate_score(state)
    otr = _on_time_ship_rate(state)

    score = 0.5 * tp + 0.3 * fr + 0.2 * otr

    return TaskResult(
        task_name="hard_throughput_scarcity",
        score=round(score, 4),
        details={
            "shipped":        len(state.shipped_orders),
            "throughput_score": round(tp, 4),
            "fill_rate":      round(fr, 4),
            "on_time_rate":   round(otr, 4),
            "total_reward":   state.total_reward,
        },
    )


ALL_TASKS = [run_task_easy, run_task_medium, run_task_hard]
