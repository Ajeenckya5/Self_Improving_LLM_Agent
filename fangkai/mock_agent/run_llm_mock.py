import json
from mock_env import build_tasks
from llm_agent import LLMAgent


class RuleBasedAgent:
    def __init__(self):
        self.actions = ["search", "click_product", "add_to_cart", "checkout", "stop"]
        self.idx = 0

    def reset(self):
        self.idx = 0

    def next_action(self, observation):
        action = self.actions[self.idx]
        self.idx += 1
        return action


def run_episode(task, agent, max_steps=None):
    if max_steps is None:
        max_steps = len(task.required_actions)

    obs = task.reset()
    agent.reset()

    trajectory = []

    for step in range(max_steps):
        action = agent.next_action(obs)
        next_obs, done, correct_step = task.step(action)

        failure_mode = None
        if not correct_step:
            if len(trajectory) > 0 and trajectory[-1]["action"] == action:
                failure_mode = "repetition_loop"
            else:
                failure_mode = "execution_error"

        trajectory.append({
            "step": step + 1,
            "observation": obs,
            "action": action,
            "next_observation": next_obs,
            "correct_step": correct_step,
            "failure_mode": failure_mode,
        })

        if failure_mode is not None:
            correct_steps = sum(t["correct_step"] for t in trajectory)
            horizon = len(task.required_actions)

            return {
                "task": task.name,
                "horizon": horizon,
                "success": False,
                "failure_mode": failure_mode,
                "steps_taken": len(trajectory),
                "completion_rate": correct_steps / horizon,
                "trajectory": trajectory,
            }

        obs = next_obs

        if done:
            correct_steps = sum(t["correct_step"] for t in trajectory)
            horizon = len(task.required_actions)

            return {
                "task": task.name,
                "horizon": horizon,
                "success": correct_step,
                "failure_mode": "none",
                "steps_taken": len(trajectory),
                "completion_rate": correct_steps / horizon,
                "trajectory": trajectory,
            }


    correct_steps = sum(t["correct_step"] for t in trajectory)
    horizon = len(task.required_actions)

    return {
        "task": task.name,
        "horizon": horizon,
        "success": False,
        "failure_mode": "timeout_or_incomplete",
        "steps_taken": len(trajectory),
        "completion_rate": correct_steps / horizon,
        "trajectory": trajectory,
    }


if __name__ == "__main__":
    tasks = build_tasks(hint_mode="weak")
    agent = LLMAgent()

    NUM_TRIALS = 10

    results = []

    for task in tasks:
        task_results = []

        for trial in range(NUM_TRIALS):
            result = run_episode(task, agent)
            result["trial"] = trial
            task_results.append(result)

        results.append({
            "task": task.name,
            "horizon": task_results[0]["horizon"],
            "trials": task_results
        })

        success_rate = sum(r["success"] for r in task_results) / NUM_TRIALS
        avg_completion = sum(r["completion_rate"] for r in task_results) / NUM_TRIALS
        avg_steps = sum(r["steps_taken"] for r in task_results) / NUM_TRIALS

        failure_counts = {}
        for r in task_results:
            mode = r.get("failure_mode", "unknown")
            failure_counts[mode] = failure_counts.get(mode, 0) + 1
        
        print(task.name, "success_rate =", success_rate)
        print(task.name, "completion_rate =", avg_completion)
        print(task.name, "avg_steps =", avg_steps)
        print(task.name, "failure_counts =", failure_counts)

    with open("mock_results.json", "w") as f:
        json.dump(results, f, indent=2)