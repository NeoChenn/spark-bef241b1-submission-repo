FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY rig ./rig
COPY policy/__init__.py policy/gemini_verifier.py ./policy/
COPY policy/fleet ./policy/fleet
COPY policy/control ./policy/control
COPY policy/gasbot/__init__.py policy/gasbot/locate.py policy/gasbot/plume.py policy/gasbot/sensors.py ./policy/gasbot/
COPY sim/output/full_rig_meta.json sim/output/leak_flanges_meta.json ./sim/output/
COPY runs_sample/fleet/fleet_state.json runs_sample/fleet/fleet_map.json ./runs_sample/fleet/
COPY runs_sample/bench/*.json ./runs_sample/bench/
RUN uv sync --frozen --no-dev
CMD ["/app/.venv/bin/uvicorn", "rig.web.server:app", "--host", "127.0.0.1", "--port", "8080"]
