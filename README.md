---
title: WarehouseFulfillmentEnv
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - supply-chain
  - logistics
  - simulation
license: mit
short_description: Real-world warehouse order-fulfillment env for AI agents
---

# 🏭 WarehouseFulfillmentEnv

A **real-world supply-chain order fulfillment** environment for training and evaluating AI agents, built to the [OpenEnv](https://openenv.dev) specification.

An agent manages a warehouse: it must pick items from inventory, fulfill customer orders, and ship them before deadlines — all under inventory constraints and time pressure.

---

## Environment Description

| Property | Value |
|----------|-------|
| Domain | Supply-chain / Logistics |
| Episode horizon | Configurable (default 50 steps) |
| Action space | Discrete union: `pick`, `ship`, `restock`, `wait` |
| Observation space | Dict (inventory, orders, deadlines, fill rates) |
| Reward | Partial-progress, shaped (see below) |
| Difficulty tiers | Easy / Medium / Hard |

### What the agent does

Each step the agent chooses one of four actions:

- **pick** — move units of a SKU from inventory into an order
- **ship** — dispatch a (possibly partial) order; scored by fill-rate and timeliness
- **restock** — add units to inventory (penalised if wasteful)
- **wait** — do nothing (penalised when work is available)

Orders have deadlines. Missing a deadline incurs a penalty. Shipping a fully-fulfilled order on time gives the maximum reward.

---

## Observation Space

```python
{
  "step":      int,          # current episode step
  "max_steps": int,          # episode horizon
  "inventory": {             # stock on hand
    "SKU-A": 12,
    "SKU-B": 7,
    ...
  },
  "orders": [
    {
      "order_id":      "ORD-000",
      "deadline_step": 30,
      "shipped":       False,
      "late":          False,
      "fill_rate":     0.5,   # 0.0 – 1.0
      "lines": [
        {"sku": "SKU-A", "qty_ordered": 4, "qty_fulfilled": 2},
        ...
      ]
    },
    ...
  ]
}
```

## Action Space

Pass actions as plain dicts:

```python
# Pick 2 units of SKU-A for order ORD-000
{"action_type": "pick", "order_id": "ORD-000", "sku": "SKU-A", "qty": 2}

# Ship order ORD-000
{"action_type": "ship", "order_id": "ORD-000"}

# Restock 5 units of SKU-B
{"action_type": "restock", "sku": "SKU-B", "qty": 5}

# Do nothing
{"action_type": "wait"}
```

## Reward Function

| Event | Reward |
|-------|--------|
| Pick progress | `+0.05 × qty × (1 + fill_rate)` |
| Ship fully fulfilled, on time | `+1.00` |
| Ship partially fulfilled | `+0.40` |
| Missed deadline | `−0.50` |
| Wasteful restock (excess units) | `−0.02 per excess unit` |
| Idle when work available | `−0.01` |
| Invalid action | `−0.02 to −0.05` |

---

## Tasks

### Task 1 — Easy: `easy_ship_2_orders`
Ship at least **2 fully-fulfilled orders on time** (3 orders, 50 steps).
Score = `min(1.0, fully_shipped_on_time / 2)`

### Task 2 — Medium: `medium_fill_and_ontime`
Achieve **≥80% mean fill-rate** AND **≥50% on-time ship rate** across 5 orders (60 steps).
Score = harmonic mean of normalised fill-rate and on-time rate.

### Task 3 — Hard: `hard_throughput_scarcity`
Ship **≥7 of 10 orders** fully and on time under **tight inventory** (80 steps).
Score = `0.5 × throughput + 0.3 × fill_rate + 0.2 × on_time_rate`

---

## Quick Start

```bash
git clone https://huggingface.co/spaces/meghanasatya/warehouse-fulfillment-env
cd warehouse-fulfillment-env
pip install -r requirements.txt
```

### Run the environment

```python
from env import WarehouseFulfillmentEnv

env = WarehouseFulfillmentEnv(n_orders=5, max_steps=50, seed=42)
obs = env.reset()

done = False
while not done:
    # your agent logic here
    action = {"action_type": "wait"}
    result = env.step(action)
    obs, reward, done = result.observation, result.reward, result.done

snap = env.state()   # WarehouseState typed model
```

### Run baseline evaluation

```bash
python baseline.py
```

Expected output (reproducible):

```
GreedyAgent
  run_task_easy     scores=[...]  mean=0.xxxx
  run_task_medium   scores=[...]  mean=0.xxxx
  run_task_hard     scores=[...]  mean=0.xxxx
```

### Run the Gradio app locally

```bash
python app.py
# → http://localhost:7860
```

### Docker

```bash
docker build -t warehouse-env .
docker run -p 7860:7860 warehouse-env
```

---

## Project Structure

```
.
├── models.py        # Typed Pydantic models (State, Action, StepResult)
├── env.py           # WarehouseFulfillmentEnv (OpenEnv API)
├── rewards.py       # Partial-progress reward functions
├── tasks.py         # 3 graded tasks with agent graders (score 0.0–1.0)
├── baseline.py      # Reproducible baseline agents + evaluation harness
├── app.py           # Gradio app (HF Spaces)
├── openenv.yaml     # OpenEnv spec manifest
├── Dockerfile       # HF Spaces container
├── requirements.txt
└── README.md
```

---

## Deploying to Hugging Face Spaces

```bash
pip install huggingface_hub python-dotenv

python deploy.py   # uses HUGGINGFACE_TOKEN from .env
```

Or manually:

```bash
huggingface-cli login
huggingface-cli repo create warehouse-fulfillment-env --type space --space-sdk docker
git remote add hf https://huggingface.co/spaces/<your-username>/warehouse-fulfillment-env
git push hf main
```

---

## License

MIT
