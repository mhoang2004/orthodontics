PROMPT 11 — Smoke test + health check

TASK:
Create production smoke test scripts

CONTEXT:
Read:
- CLAUDE.md
- backend API contract
- Triton model contract

GOAL:
Verify:
- Triton server boots
- model loads successfully
- FastAPI connects to Triton
- inference request succeeds

STRICT RULES:
- testing only
- no source refactor

OUTPUT:
Create:
- smoke_test.py
- health_check.sh
- TESTING.md