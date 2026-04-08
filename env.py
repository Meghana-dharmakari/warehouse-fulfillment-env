"""
env.py — WarehouseFulfillmentEnv

Real-world task: a warehouse receives customer orders and must pick items
from inventory, then ship orders before their deadlines.

OpenEnv API
-----------
    reset()  → dict          initial observation
    step(action) → StepResult
    state()  → WarehouseState
"""

from __future__ import annotations
import copy, random
from typing import Any

from models import (
    Action, Order, OrderLine, PickAction, RestockAction,
    ShipAction, WaitAction, WarehouseState, StepResult,
)
from rewards import (
    deadline_miss_penalty, idle_penalty, pick_reward,
    restock_reward, ship_reward,
)

# ── default catalogue ──────────────────────────────────────────────────────────
DEFAULT_SKUS = ["SKU-A", "SKU-B", "SKU-C", "SKU-D", "SKU-E"]


class WarehouseFulfillmentEnv:
    """
    OpenEnv-compatible warehouse order-fulfillment environment.

    Observation space
    -----------------
    {
      "step":      int,
      "max_steps": int,
      "inventory": {sku: qty, ...},
      "orders": [
        {
          "order_id": str,
          "lines": [{"sku": str, "qty_ordered": int, "qty_fulfilled": int}],
          "deadline_step": int,
          "shipped": bool
        }, ...
      ]
    }

    Action space  (pass as dict or typed model)
    ------------
    {"action_type": "pick",    "order_id": str, "sku": str, "qty": int}
    {"action_type": "ship",    "order_id": str}
    {"action_type": "restock", "sku": str, "qty": int}
    {"action_type": "wait"}
    """

    metadata = {
        "name": "WarehouseFulfillmentEnv-v1",
        "version": "1.0.0",
        "task": "Supply-chain order fulfillment",
        "action_types": ["pick", "ship", "restock", "wait"],
        "reward_range": (-float("inf"), float("inf")),
    }

    def __init__(
        self,
        skus: list[str] | None = None,
        n_orders: int = 5,
        max_steps: int = 50,
        seed: int | None = None,
    ):
        self.skus      = skus or DEFAULT_SKUS
        self.n_orders  = n_orders
        self.max_steps = max_steps
        self._rng      = random.Random(seed)
        self._state    = WarehouseState()

    # ── OpenEnv API ────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reset environment and return initial observation."""
        inventory = {sku: self._rng.randint(5, 20) for sku in self.skus}
        orders = [
            self._make_order(i, inventory)
            for i in range(self.n_orders)
        ]
        self._state = WarehouseState(
            step=0,
            max_steps=self.max_steps,
            inventory=inventory,
            orders=orders,
        )
        return self._observe()

    def step(self, action: dict | Action) -> StepResult:
        """Advance the environment by one step."""
        if self._state.done:
            raise RuntimeError("Episode finished — call reset() first.")

        action = self._parse_action(action)
        reward, info = self._apply_action(action)

        # advance clock and check deadlines
        self._state.step += 1
        deadline_reward = self._check_deadlines()
        reward += deadline_reward

        self._state.total_reward += reward
        self._state.done = (
            self._state.step >= self._state.max_steps
            or self._all_done()
        )

        return StepResult(
            observation=self._observe(),
            reward=round(reward, 4),
            done=self._state.done,
            info=info,
        )

    def state(self) -> WarehouseState:
        """Return a deep copy of the current environment state."""
        return copy.deepcopy(self._state)

    # ── internal helpers ───────────────────────────────────────────────────────

    def _observe(self) -> dict:
        s = self._state
        return {
            "step":      s.step,
            "max_steps": s.max_steps,
            "inventory": dict(s.inventory),
            "orders": [
                {
                    "order_id":      o.order_id,
                    "deadline_step": o.deadline_step,
                    "shipped":       o.shipped,
                    "late":          o.late,
                    "fill_rate":     round(o.total_fill_rate, 3),
                    "lines": [
                        {
                            "sku":           l.sku,
                            "qty_ordered":   l.qty_ordered,
                            "qty_fulfilled": l.qty_fulfilled,
                        }
                        for l in o.lines
                    ],
                }
                for o in s.orders
            ],
        }

    def _make_order(self, idx: int, inventory: dict[str, int]) -> Order:
        n_lines  = self._rng.randint(1, min(3, len(self.skus)))
        skus     = self._rng.sample(self.skus, n_lines)
        lines    = [
            OrderLine(
                sku=sku,
                qty_ordered=self._rng.randint(1, max(1, inventory[sku] // 2)),
            )
            for sku in skus
        ]
        deadline = self._rng.randint(
            self.max_steps // 4,
            self.max_steps - 1,
        )
        return Order(
            order_id=f"ORD-{idx:03d}",
            lines=lines,
            deadline_step=deadline,
        )

    def _parse_action(self, action: dict | Action) -> Action:
        if isinstance(action, dict):
            t = action.get("action_type", "wait")
            if t == "pick":
                return PickAction(**action)
            if t == "ship":
                return ShipAction(**action)
            if t == "restock":
                return RestockAction(**action)
            return WaitAction()
        return action

    def _apply_action(self, action: Action) -> tuple[float, dict]:
        s = self._state
        info: dict[str, Any] = {"action": action.action_type}

        if isinstance(action, PickAction):
            order = self._find_order(action.order_id)
            if order is None or order.shipped:
                info["error"] = "invalid order"
                return -0.05, info
            line = next((l for l in order.lines if l.sku == action.sku), None)
            if line is None:
                info["error"] = "sku not in order"
                return -0.05, info
            available = s.inventory.get(action.sku, 0)
            qty = min(action.qty, line.remaining, available)
            if qty <= 0:
                info["error"] = "nothing to pick"
                return -0.02, info
            s.inventory[action.sku] = available - qty
            line.qty_fulfilled += qty
            r = pick_reward(qty, order)
            info.update({"picked": qty, "sku": action.sku, "order": action.order_id})
            return r, info

        if isinstance(action, ShipAction):
            order = self._find_order(action.order_id)
            if order is None or order.shipped:
                info["error"] = "invalid or already shipped"
                return -0.05, info
            r = ship_reward(order, s.step)
            order.shipped = True
            s.shipped_orders.append(order)
            info.update({"shipped": action.order_id, "fill_rate": order.total_fill_rate})
            return r, info

        if isinstance(action, RestockAction):
            r = restock_reward(action.sku, action.qty, s)
            s.inventory[action.sku] = s.inventory.get(action.sku, 0) + action.qty
            info.update({"restocked": action.sku, "qty": action.qty})
            return r, info

        # WaitAction
        r = idle_penalty(s)
        info["idle"] = True
        return r, info

    def _check_deadlines(self) -> float:
        """Penalise orders that just missed their deadline."""
        penalty = 0.0
        for order in self._state.orders:
            if (
                not order.shipped
                and not order.late
                and self._state.step > order.deadline_step
            ):
                order.late = True
                penalty += deadline_miss_penalty(order)
        return penalty

    def _find_order(self, order_id: str) -> Order | None:
        return next((o for o in self._state.orders if o.order_id == order_id), None)

    def _all_done(self) -> bool:
        return all(o.shipped or o.late for o in self._state.orders)
