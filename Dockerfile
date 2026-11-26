FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --locked --extra api --no-dev --no-install-project --no-cache
COPY ml ./ml
COPY web ./web
COPY --from=frontend /frontend/dist ./frontend/dist
RUN useradd --create-home mechanic && install -d -o mechanic -g mechanic /state
USER mechanic
CMD ["uvicorn", "web.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
