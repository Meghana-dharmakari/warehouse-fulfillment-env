"""
baseline.py — reproducible baseline agents for WarehouseFulfillmentEnv.

Agents
------
  random_agent   : uniform random valid action
  greedy_agent   : always pick → ship when full, restock if empty

Run
---
    python baseline.py
"""

from __future__ import annotations
import json, random
from tasks import run_task_easy, run_task_medium, run_task_hard, ALL_TASKS

SEED = 42


# ── Random agent ───────────────────────────────────────────────────────────────

_rng = random.Random(SEED)

def random_agent(obs: dict) -> dict:
    """Picks a random valid action from the current observation."""
    orders = [o for o in obs["orders"] if not o["shipped"]]
    if not orders:
        return {"action_type": "wait"}

    choice = _rng.random()

    if choice < 0.6 and orders:
        order = _rng.choice(orders)
        pickable = [
            l for l in order["lines"]
            if l["qty_fulfilled"] < l["qty_ordered"]
            and obs["inventory"].get(l["sku"], 0) > 0
        ]
        if pickable:
            line = _rng.choice(pickable)
            return {
                "action_type": "pick",
                "order_id":    order["order_id"],
                "sku":         line["sku"],
                "qty":         _rng.randint(1, max(1, line["qty_ordered"] - line["qty_fulfilled"])),
            }

    if choice < 0.75:
        shippable = [o for o in orders if o["fill_rate"] >= 0.5]
        if shippable:
            return {"action_type": "ship", "order_id": _rng.choice(shippable)["order_id"]}

    if choice < 0.85:
        sku = _rng.choice(list(obs["inventory"].keys()))
        return {"action_type": "restock", "sku": sku, "qty": _rng.randint(1, 5)}

    return {"action_type": "wait"}


# ── Greedy agent ───────────────────────────────────────────────────────────────

def greedy_agent(obs: dict) -> dict:
    """
    Heuristic:
      1. Ship any fully-fulfilled order.
      2. Pick the most-urgent (earliest deadline) order's missing items.
      3. Restock if a needed SKU is out of stock.
      4. Wait.
    """
    orders = [o for o in obs["orders"] if not o["shipped"]]
    inventory = obs["inventory"]

    # 1. ship fully fulfilled
    for order in orders:
        if order["fill_rate"] == 1.0:
            return {"action_type": "ship", "order_id": order["order_id"]}

    # 2. sort by deadline urgency
    pending = sorted(orders, key=lambda o: o["deadline_step"])
    for order in pending:
        for line in order["lines"]:
            remaining = line["qty_ordered"] - line["qty_fulfilled"]
            if remaining <= 0:
                continue
            stock = inventory.get(line["sku"], 0)
            if stock > 0:
                return {
                    "action_type": "pick",
                    "order_id":    order["order_id"],
                    "sku":         line["sku"],
                    "qty":         min(remaining, stock),
                }
            # 3. restock if needed
            return {"action_type": "restock", "sku": line["sku"], "qty": remaining}

    return {"action_type": "wait"}


# ── Evaluation harness ─────────────────────────────────────────────────────────

def evaluate(agent_fn, name: str, n_seeds: int = 5) -> dict:
    results = {}
    for task_fn in ALL_TASKS:
        scores = [task_fn(agent_fn, seed=s).score for s in range(n_seeds)]
        avg = round(sum(scores) / len(scores), 4)
        results[task_fn.__name__] = {"scores": scores, "mean": avg}
    return {"agent": name, "results": results}


if __name__ == "__main__":
    print("=" * 60)
    print("WarehouseFulfillmentEnv — Baseline Evaluation")
    print(f"Seeds: 0–{4}  |  Reproducible with SEED={SEED}")
    print("=" * 60)

    for agent, label in [(random_agent, "RandomAgent"), (greedy_agent, "GreedyAgent")]:
        report = evaluate(agent, label)
        print(f"\n{label}")
        for task, data in report["results"].items():
            print(f"  {task:<35} scores={data['scores']}  mean={data['mean']:.4f}")

    print("\nDone.")
