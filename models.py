"""models.py — typed data models for WarehouseFulfillmentEnv."""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class OrderLine(BaseModel):
    sku: str
    qty_ordered: int
    qty_fulfilled: int = 0

    @property
    def remaining(self) -> int:
        return self.qty_ordered - self.qty_fulfilled

    @property
    def fill_rate(self) -> float:
        return self.qty_fulfilled / self.qty_ordered if self.qty_ordered else 1.0


class Order(BaseModel):
    order_id: str
    lines: list[OrderLine]
    deadline_step: int          # episode step by which order must ship
    shipped: bool = False
    late: bool = False

    @property
    def total_fill_rate(self) -> float:
        if not self.lines:
            return 1.0
        return sum(l.fill_rate for l in self.lines) / len(self.lines)

    @property
    def fully_fulfilled(self) -> bool:
        return all(l.remaining == 0 for l in self.lines)


class WarehouseState(BaseModel):
    step: int = 0
    max_steps: int = 50
    inventory: dict[str, int] = Field(default_factory=dict)
    orders: list[Order] = Field(default_factory=list)
    shipped_orders: list[Order] = Field(default_factory=list)
    total_reward: float = 0.0
    done: bool = False


class PickAction(BaseModel):
    action_type: Literal["pick"] = "pick"
    order_id: str
    sku: str
    qty: int = Field(ge=1)


class ShipAction(BaseModel):
    action_type: Literal["ship"] = "ship"
    order_id: str


class RestockAction(BaseModel):
    action_type: Literal["restock"] = "restock"
    sku: str
    qty: int = Field(ge=1)


class WaitAction(BaseModel):
    action_type: Literal["wait"] = "wait"


Action = PickAction | ShipAction | RestockAction | WaitAction


class StepResult(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict
