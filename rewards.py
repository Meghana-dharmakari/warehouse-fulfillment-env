"""rewards.py — partial-progress reward shaping for WarehouseFulfillmentEnv."""

from __future__ import annotations
from models import Order, WarehouseState


# Reward weights
W_PICK          =  0.05   # per unit picked toward an order
W_SHIP_FULL     =  1.00   # ship a fully-fulfilled order on time
W_SHIP_PARTIAL  =  0.40   # ship a partially-fulfilled order on time
W_LATE_PENALTY  = -0.50   # order misses deadline
W_OVERSTOCK     = -0.02   # per unit restocked beyond current need (waste)
W_IDLE          = -0.01   # wasted step (wait with pending work)


def pick_reward(qty_picked: int, order: Order) -> float:
    """Incremental reward for picking units toward an order."""
    return W_PICK * qty_picked * (1.0 + order.total_fill_rate)


def ship_reward(order: Order, current_step: int) -> float:
    """Reward for shipping; penalises partial shipments."""
    on_time = current_step <= order.deadline_step
    base = W_SHIP_FULL if order.fully_fulfilled else W_SHIP_PARTIAL
    if not on_time:
        base += W_LATE_PENALTY
    return base


def deadline_miss_penalty(order: Order) -> float:
    return W_LATE_PENALTY


def restock_reward(sku: str, qty: int, state: WarehouseState) -> float:
    """Small penalty if restocking a SKU that already has enough stock."""
    needed = sum(
        l.remaining
        for o in state.orders
        for l in o.lines
        if l.sku == sku and not o.shipped
    )
    excess = max(0, (state.inventory.get(sku, 0) + qty) - needed)
    return W_OVERSTOCK * excess


def idle_penalty(state: WarehouseState) -> float:
    """Penalise waiting when there is actionable work."""
    has_work = any(
        not o.shipped and o.total_fill_rate < 1.0
        for o in state.orders
    )
    return W_IDLE if has_work else 0.0
