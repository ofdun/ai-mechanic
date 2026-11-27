FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir playwright==1.55.0 \
    && playwright install --with-deps --only-shell chromium \
    && rm -rf /var/lib/apt/lists/*
