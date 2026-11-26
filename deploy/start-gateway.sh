#!/bin/sh
set -eu

if [ -n "${STAND_PASSWORD:-}" ]; then
    if [ -n "${STAND_PASSWORD_HASH:-}" ]; then
        echo "Set only one of STAND_PASSWORD and STAND_PASSWORD_HASH" >&2
        exit 1
    fi
    STAND_PASSWORD_HASH=$(printf '%s' "$STAND_PASSWORD" | caddy hash-password)
    export STAND_PASSWORD_HASH
    unset STAND_PASSWORD
fi

: "${STAND_PASSWORD_HASH:?Set STAND_PASSWORD or STAND_PASSWORD_HASH in .env}"
exec caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
