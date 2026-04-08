"""
run.py — train ReinforceAgent on BalanceEnv and print progress.

Usage
-----
    python run.py
"""

from env   import BalanceEnv
from agent import ReinforceAgent

EPISODES    = 500
LOG_EVERY   = 50
SEED        = 42


def run_episode(env: BalanceEnv, agent: ReinforceAgent):
    """Collect one full episode trajectory."""
    obs        = env.reset()
    trajectory = []
    done       = False

    while not done:
        action          = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)
        trajectory.append((obs, action, reward))
        obs = next_obs

    return trajectory


def main():
    env   = BalanceEnv(seed=SEED)
    agent = ReinforceAgent(obs_dim=4, n_actions=2, lr=1e-2, gamma=0.99, seed=SEED)

    recent_returns: list[float] = []

    for ep in range(1, EPISODES + 1):
        trajectory = run_episode(env, agent)
        loss       = agent.update(trajectory)
        ep_return  = sum(r for _, _, r in trajectory)

        recent_returns.append(ep_return)
        if len(recent_returns) > LOG_EVERY:
            recent_returns.pop(0)

        if ep % LOG_EVERY == 0:
            avg = sum(recent_returns) / len(recent_returns)
            # also show current env snapshot via state()
            snap = env.state()
            print(
                f"Episode {ep:>4} | "
                f"avg_return={avg:>7.1f} | "
                f"loss={loss:>7.4f} | "
                f"env.state()={snap}"
            )

    print("\nTraining complete.")
    print("Final env snapshot:", env.state())


if __name__ == "__main__":
    main()
