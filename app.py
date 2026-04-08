# """
# app.py — Gradio interface for WarehouseFulfillmentEnv on Hugging Face Spaces.

# Exposes:
#   • Interactive episode runner (greedy or random agent)
#   • Task benchmark panel (scores all 3 tasks)
# """

# import json
# import gradio as gr
# from env import WarehouseFulfillmentEnv
# from baseline import greedy_agent, random_agent, evaluate
# from tasks import run_task_easy, run_task_medium, run_task_hard


# # ── Episode runner ─────────────────────────────────────────────────────────────

# def run_episode(agent_name: str, n_orders: int, max_steps: int, seed: int):
#     agent = greedy_agent if agent_name == "Greedy" else random_agent
#     env = WarehouseFulfillmentEnv(n_orders=n_orders, max_steps=max_steps, seed=seed)
#     obs = env.reset()
#     done = False
#     log = []

#     while not done:
#         action = agent(obs)
#         result = env.step(action)
#         log.append({
#             "step":   obs["step"],
#             "action": action,
#             "reward": result.reward,
#             "done":   result.done,
#         })
#         obs, done = result.observation, result.done

#     state = env.state()
#     summary = {
#         "total_reward":   round(state.total_reward, 4),
#         "steps_taken":    state.step,
#         "orders_shipped": len(state.shipped_orders),
#         "orders_total":   len(state.orders),
#         "fill_rates": {
#             o.order_id: round(o.total_fill_rate, 3)
#             for o in state.orders + state.shipped_orders
#         },
#     }
#     return json.dumps(summary, indent=2), json.dumps(log[-10:], indent=2)


# # ── Benchmark panel ────────────────────────────────────────────────────────────

# def run_benchmark(agent_name: str, n_seeds: int):
#     agent = greedy_agent if agent_name == "Greedy" else random_agent
#     report = evaluate(agent, agent_name, n_seeds=int(n_seeds))
#     rows = []
#     for task, data in report["results"].items():
#         rows.append([task, str(data["scores"]), f"{data['mean']:.4f}"])
#     return rows


# # ── Gradio UI ──────────────────────────────────────────────────────────────────

# with gr.Blocks(title="WarehouseFulfillmentEnv") as demo:
#     gr.Markdown(
#         """
#         # 🏭 WarehouseFulfillmentEnv
#         **Real-world supply-chain order fulfillment environment for AI agents.**

#         An agent manages a warehouse: picking items, fulfilling orders, and shipping
#         before deadlines — with partial-progress rewards and 3 difficulty tiers.
#         """
#     )

#     with gr.Tab("Episode Runner"):
#         with gr.Row():
#             agent_sel  = gr.Dropdown(["Greedy", "Random"], value="Greedy", label="Agent")
#             n_orders   = gr.Slider(2, 10, value=5, step=1, label="Orders")
#             max_steps  = gr.Slider(20, 100, value=50, step=5, label="Max Steps")
#             seed_input = gr.Number(value=42, label="Seed", precision=0)
#         run_btn = gr.Button("▶ Run Episode")
#         with gr.Row():
#             summary_out = gr.Code(label="Episode Summary", language="json")
#             log_out     = gr.Code(label="Last 10 Steps",   language="json")
#         run_btn.click(
#             run_episode,
#             inputs=[agent_sel, n_orders, max_steps, seed_input],
#             outputs=[summary_out, log_out],
#         )

#     with gr.Tab("Task Benchmark"):
#         with gr.Row():
#             bench_agent = gr.Dropdown(["Greedy", "Random"], value="Greedy", label="Agent")
#             n_seeds_sl  = gr.Slider(1, 10, value=5, step=1, label="Seeds")
#         bench_btn = gr.Button("📊 Run Benchmark")
#         bench_table = gr.Dataframe(
#             headers=["Task", "Scores", "Mean Score"],
#             label="Results",
#         )
#         bench_btn.click(
#             run_benchmark,
#             inputs=[bench_agent, n_seeds_sl],
#             outputs=[bench_table],
#         )

#     with gr.Tab("API Reference"):
#         gr.Markdown(
#             """
#             ## OpenEnv API

#             ```python
#             from env import WarehouseFulfillmentEnv

#             env = WarehouseFulfillmentEnv(n_orders=5, max_steps=50, seed=42)
#             obs  = env.reset()          # → dict
#             result = env.step({         # → StepResult
#                 "action_type": "pick",
#                 "order_id": "ORD-000",
#                 "sku": "SKU-A",
#                 "qty": 2,
#             })
#             snap = env.state()          # → WarehouseState
#             ```

#             ## Action Types
#             | action_type | Required fields |
#             |-------------|----------------|
#             | `pick`      | order_id, sku, qty |
#             | `ship`      | order_id |
#             | `restock`   | sku, qty |
#             | `wait`      | — |

#             ## Reward Signals
#             | Event | Reward |
#             |-------|--------|
#             | Pick progress | +0.05 × qty × (1 + fill_rate) |
#             | Ship full on-time | +1.00 |
#             | Ship partial | +0.40 |
#             | Late deadline | −0.50 |
#             | Overstock waste | −0.02/unit |
#             | Idle with work | −0.01 |
#             """
#         )

# demo.launch()




from fastapi import FastAPI
import gradio as gr
from env import WarehouseFulfillmentEnv

app = FastAPI()

# Global env instance
env = WarehouseFulfillmentEnv(n_orders=5, max_steps=50, seed=42)

# ── REQUIRED API ENDPOINTS ─────────────────────────

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs}

@app.post("/step")
def step(action: dict):
    result = env.step(action)
    return {
        "observation": result.observation,
        "reward": result.reward,
        "done": result.done
    }

@app.get("/state")
def state():
    s = env.state()
    return {
        "step": s.step,
        "total_reward": s.total_reward
    }

# ── GRADIO UI (your existing UI) ───────────────────

def run_episode():
    env.reset()
    return "Running..."

demo = gr.Interface(fn=run_episode, inputs=[], outputs="text")

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(app, demo, path="/")