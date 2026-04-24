PROMPT 7 — Create config.pbtxt

TASK:
Create triton_model_repository/orthodontics_pipeline/config.pbtxt

CONTEXT:
Read only:
- CLAUDE.md

GOAL:
Use Input-Output Tensor Spec from CLAUDE.md

STRICT RULES:
- no guessing
- exact tensor definitions only
- production-ready Triton config
- batch policy must be explicit

OUTPUT:
Overwrite config.pbtxt