FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY app/frontend/package*.json ./
RUN npm ci
COPY app/frontend/ ./
RUN npm run build

FROM golang:1.25-bookworm AS backend
WORKDIR /src
COPY app/backend/go.* ./
RUN go mod download
COPY app/backend/ ./
RUN go test ./... && go build -trimpath -o /out/mechanic .

FROM python:3.12-slim-bookworm AS agent
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --locked --extra api --no-dev --no-install-project --no-cache
COPY ml ./ml
RUN useradd --create-home mechanic && install -d -o mechanic -g mechanic /state
USER mechanic
CMD ["uvicorn","ml.worker:create_app","--factory","--host","0.0.0.0","--port","8000"]

FROM debian:bookworm-slim AS web
RUN useradd --create-home mechanic && install -d -o mechanic -g mechanic /state
COPY --from=backend /out/mechanic /usr/local/bin/mechanic
COPY --from=frontend /frontend/dist /app/frontend/dist
USER mechanic
ENTRYPOINT ["mechanic"]
