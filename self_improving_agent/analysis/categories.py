"""Failure categories for classification and retrieval tagging."""

FAILURE_CATEGORIES = [
    "planning_error",               # Wrong sequence of steps, missing steps
    "memory_limitation",            # Forgot earlier context or constraints
    "instruction_misinterpretation",# Misunderstood the task
    "environmental_change",         # Assumed wrong state (file not there, wrong schema)
    "false_assumption",             # Incorrect assumption about data or system
    "repeated_action",              # Same action repeated with no progress
    "circular_loop",                # Action sequence that repeats as a subsequence
    "context_truncation",           # Observation exceeded context window
    "tool_misuse",                  # Malformed tool calls or wrong tool for the task
    "incorrect_reasoning",          # Contradicted a prior successful observation
]

CATEGORY_DESCRIPTIONS = {
    "planning_error": "The agent chose the wrong sequence of steps or omitted critical steps",
    "memory_limitation": "The agent forgot earlier context, constraints, or partial results",
    "instruction_misinterpretation": "The agent misunderstood what the task required",
    "environmental_change": "The agent assumed wrong environment state (e.g. file paths, schema)",
    "false_assumption": "The agent made an incorrect assumption about data or system behavior",
    "repeated_action": "The agent repeated the same action ≥3 times without progress",
    "circular_loop": "The agent executed a repeated sequence of actions in a loop",
    "context_truncation": "Observations exceeded the context window causing information loss",
    "tool_misuse": "The agent produced malformed tool calls or used wrong tool for the task",
    "incorrect_reasoning": "The agent re-attempted actions that had already succeeded",
}
