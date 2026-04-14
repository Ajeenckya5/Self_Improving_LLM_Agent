"""
Self-Improving LLM Agent for Long-Horizon Tasks.

Architecture:
  Task Dataset → LLM Agent (Plan-Act Loop) → Execution Trace
      → Failure Analyzer → Strategy Generator → Strategy Memory
      → Strategy-Guided Execution (improved agent on next task)
"""

__version__ = "0.1.0"
