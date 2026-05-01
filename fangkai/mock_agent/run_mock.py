import json
from mock_env import build_tasks


class RuleBasedAgent:
    def __init__(self):
        self.actions = []
        self.idx = 0

    def reset(self, task=None):
        self.idx = 0
        if task is not None:
            self.actions = task.required_actions

    def next_action(self, observation):
        action = self.actions[self.idx]
        self.idx += 1
        return action


def run_episode(task, agent, max_steps=None):
    if max_steps is None:
        max_steps = len(task.required_actions)
        
    obs = task.reset()
    agent.reset(task)

    trajectory = []

    for step in range(max_steps):
        action = agent.next_action(obs)
        next_obs, done, correct_step = task.step(action)

        trajectory.append({
            "step": step + 1,
            "observation": obs,
            "action": action,
            "next_observation": next_obs,
            "correct_step": correct_step,
        })

        obs = next_obs

        if done:
            return {
                "task": task.name,
                "horizon": len(task.required_actions),
                "success": correct_step,
                "trajectory": trajectory,
            }

    return {
        "task": task.name,
        "horizon": len(task.required_actions),
        "success": False,
        "trajectory": trajectory,
    }


if __name__ == "__main__":
    tasks = build_tasks()
    agent = RuleBasedAgent()

    results = []

    for task in tasks:
        result = run_episode(task, agent)
        results.append(result)
        print(task.name, "success =", result["success"])

    with open("mock_results.json", "w") as f:
        json.dump(results, f, indent=2)