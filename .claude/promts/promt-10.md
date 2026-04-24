PROMPT 10 — Docker + Compose production setup

TASK:
Create production Docker deployment

CONTEXT:
Read:
- CLAUDE.md
- Docker-related files only

GOAL:
Support:
- FastAPI
- Triton Server
- GPU runtime if available

STRICT RULES:
- no code refactor
- infra only

OUTPUT:
Create/update:
- Dockerfile
- docker-compose.yml
- startup scripts if needed