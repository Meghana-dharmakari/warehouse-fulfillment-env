from fastapi import FastAPI
import gradio as gr
from env import WarehouseFulfillmentEnv

app = FastAPI()

env = WarehouseFulfillmentEnv(n_orders=5, max_steps=50, seed=42)

# ✅ REQUIRED APIs

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

# ✅ SIMPLE UI (can keep minimal)

def hello():
    return "Warehouse Env Running"

demo = gr.Interface(fn=hello, inputs=[], outputs="text")

# ❗ IMPORTANT: mount instead of launch
app = gr.mount_gradio_app(app, demo, path="/")