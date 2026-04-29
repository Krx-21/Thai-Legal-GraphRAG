"""Regression tests for Thai Legal GraphRAG.

Run:  .venv\\Scripts\\python.exe -m pytest tests/ -v

Two test layers:

1. Pure-unit tests (fast, no KG): regex extraction, citation F1, k-fold split.
2. Snapshot tests (slow, ~30s): load KG once and verify a small set of frozen
   questions still retrieve the expected sections above threshold. These are
   the canary tests that catch retrieval regressions when boost keywords,
   tokenization, or scoring are changed.

Thresholds are intentionally loose (e.g. F1 >= 0.4 instead of an exact value)
so day-to-day tuning doesn't break tests, but a hard regression in retrieval
quality will trip them.
"""
