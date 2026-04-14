"""Pytest configuration — force mock LLM for all tests."""
import os
os.environ["MOCK_LLM"] = "1"
