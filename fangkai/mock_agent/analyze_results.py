import json

with open("mock_results.json") as f:
    data = json.load(f)

print("\n=== Horizon Summary ===")

for task in data:
    horizon = task["horizon"]
    trials = task["trials"]

    success_rate = sum(t["success"] for t in trials) / len(trials)
    completion_rate = sum(t["completion_rate"] for t in trials) / len(trials)
    avg_steps = sum(t["steps_taken"] for t in trials) / len(trials)

    failure_counts = {}
    for t in trials:
        mode = t.get("failure_mode", "unknown")
        failure_counts[mode] = failure_counts.get(mode, 0) + 1

    print(
        f"H={horizon} | success_rate={success_rate:.2f} "
        f"| completion={completion_rate:.2f} | avg_steps={avg_steps:.2f} "
        f"| failures={failure_counts}"
    )