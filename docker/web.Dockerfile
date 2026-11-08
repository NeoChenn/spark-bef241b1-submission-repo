FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY rig ./rig
COPY policy/fleet ./policy/fleet
COPY sim/output/full_rig_meta.json sim/output/leak_flanges_meta.json ./sim/output/
RUN uv sync --frozen --no-dev
CMD ["/app/.venv/bin/uvicorn", "rig.web.server:app", "--host", "127.0.0.1", "--port", "8080"]
