PROMPT 9 — Refactor FastAPI backend integration

TASK:
Refactor backend/app.py for Triton integration

CONTEXT:
Read only:
- CLAUDE.md
- backend/app.py

GOAL:
Replace direct model inference
with Triton Inference Server calls

STRICT RULES:
- only edit app.py
- no Docker changes
- no requirements.txt changes
- preserve existing API contract

OUTPUT:
1. overwrite app.py
2. create DONE_BACKEND.md