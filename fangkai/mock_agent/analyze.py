import json
import matplotlib.pyplot as plt

with open("mock_results.json") as f:
    data = json.load(f)

horizons = []
success_rates = []
completion_rates = []

for task in data:
    h = task["horizon"]
    trials = task["trials"]

    success = sum(t["success"] for t in trials) / len(trials)
    completion = sum(t["completion_rate"] for t in trials) / len(trials)

    horizons.append(h)
    success_rates.append(success)
    completion_rates.append(completion)

plt.plot(horizons, success_rates, marker='o', label="Success Rate")
plt.plot(horizons, completion_rates, marker='x', label="Completion Rate")

plt.xlabel("Horizon")
plt.ylabel("Performance")
plt.title("LLM Agent Scaling Behavior")
plt.legend()
plt.grid()

plt.show()