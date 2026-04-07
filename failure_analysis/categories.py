"""Failure categories for LLM classification."""

FAILURE_CATEGORIES = [
    "planning_error",       # Wrong sequence of steps, missing steps
    "memory_limitation",    # Forgot earlier context or constraints
    "instruction_misinterpretation",  # Misunderstood the task
    "environmental_change", # Assumed wrong state (file not there, wrong schema)
    "false_assumption",     # Incorrect assumption about data or system
]

CATEGORY_DESCRIPTIONS = {
    "planning_error": "The agent chose the wrong sequence of steps or omitted critical steps",
    "memory_limitation": "The agent forgot earlier context, constraints, or partial results",
    "instruction_misinterpretation": "The agent misunderstood what the task required",
    "environmental_change": "The agent assumed wrong environment state (e.g. file paths, schema)",
    "false_assumption": "The agent made an incorrect assumption about data or system behavior",
}
