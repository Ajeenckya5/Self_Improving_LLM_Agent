# Terminal-Bench / Harbor Runbook

This branch contains everything needed to test the self-improving agent on Terminal-Bench.

## Branch

Use:

```bash
git checkout AJ_EXPERIMENTS
```

This branch reports only the named contributor integrations:

- Akash: `origin/member1`
- Bigye: `origin/member_3`
- Pratik: `origin/639_member2`
- Fangkai: `origin/fangkai`

`main` is intentionally not listed as a contributor branch in the report/graphs.

## Setup

```bash
source .venv/bin/activate
pip install -r self_improving_agent/requirements.txt
```

Docker must be running before executing Harbor or Terminal-Bench tasks.

Check:

```bash
docker info
harbor --help
```

If `harbor` is missing:

```bash
pip install "harbor>=0.6.4"
```

## Harbor Smoke Test

First verify Harbor can run the official oracle solution:

```bash
harbor run \
  -d terminal-bench/terminal-bench-2 \
  -a oracle \
  -l 1 \
  -n 1 \
  --job-name aj-oracle-smoke \
  --yes
```

## Run Our Agent on Terminal-Bench 2.0

Set your model key first. For the current config profile:

```bash
export XAI_API_KEY="<your-key>"
```

Run one task as a smoke test:

```bash
harbor run \
  -d terminal-bench/terminal-bench-2 \
  --agent-import-path self_improving_agent.integrations.harbor_agent:SelfImprovingHarborAgent \
  --ak config_path="$(pwd)/self_improving_agent/config.yaml" \
  --ak profile=xai \
  --ak max_steps=50 \
  --ak command_timeout_sec=180 \
  -l 1 \
  -n 1 \
  --job-name aj-agent-smoke \
  --yes
```

Run a larger local evaluation:

```bash
harbor run \
  -d terminal-bench/terminal-bench-2 \
  --agent-import-path self_improving_agent.integrations.harbor_agent:SelfImprovingHarborAgent \
  --ak config_path="$(pwd)/self_improving_agent/config.yaml" \
  --ak profile=xai \
  --ak max_steps=80 \
  --ak command_timeout_sec=240 \
  -l 25 \
  -n 4 \
  --job-name aj-agent-tbench-25 \
  --yes
```

## Upload Results

To upload directly after the run:

```bash
harbor run \
  -d terminal-bench/terminal-bench-2 \
  --agent-import-path self_improving_agent.integrations.harbor_agent:SelfImprovingHarborAgent \
  --ak config_path="$(pwd)/self_improving_agent/config.yaml" \
  --ak profile=xai \
  --ak max_steps=80 \
  --ak command_timeout_sec=240 \
  -l 25 \
  -n 4 \
  --job-name aj-agent-tbench-25 \
  --upload \
  --private \
  --yes
```

Use `--public` only when the result is ready to share.

## Legacy Terminal-Bench CLI

The branch also includes the older `tb` runner:

```bash
python main.py terminalbench --print-command --task-id hello-world --n-tasks 1 --profile xai
```

For the Harbor docs you linked, prefer the `harbor run ...` commands above.

## Files Added for Terminal-Bench

- `self_improving_agent/integrations/harbor_agent.py`
- `self_improving_agent/integrations/terminal_bench_agent.py`
- `self_improving_agent/experiments/run_terminal_bench.py`
- `self_improving_agent/prompts/terminal_bench_agent.txt`
- `self_improving_agent/tests/test_terminal_bench_integration.py`
