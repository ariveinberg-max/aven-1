#!/usr/bin/env bash
# Auto-restart wrapper for unattended train.py runs (e.g. a rented GPU pod).
#
# Usage: same args you'd give train.py directly, e.g.:
#   ./run_resilient.sh --data data/training.txt --resume --steps 5000 \
#     --expect-params 58424832 --device cuda --wandb
#
# On a crash/disconnect (any exit code except a clean 0 finish or an
# argparse config-error exit of 2), it waits and retries with --resume
# appended so it picks up from the last autosaved checkpoint. Exit 2 means
# train.py itself rejected the invocation (bad flags, --expect-params
# mismatch, etc.) — retrying won't fix that, so the wrapper stops instead
# of silently retrying a broken config forever on a paid GPU.
#
# Set CONTINUOUS=1 to keep chaining more --resume chunks after each clean
# finish (train.py caps --steps at 10000 per invocation) instead of
# stopping — useful to keep a rented GPU productive for as long as its
# budget lasts, rather than idling (and still billing) after an early
# finish. There's no step-count ceiling in this mode; stop it yourself
# (Ctrl+C, or when the account runs out of funds) once you've had enough.

set -u
cd "$(dirname "$0")"

if [ -n "${PYTHON:-}" ]; then
    :  # explicit override (e.g. Windows, where PATH resolution of
       # python/python3 is unreliable -- see the App execution alias issue)
elif [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
else
    PYTHON="$(command -v python3 || command -v python)"
fi

CONTINUOUS="${CONTINUOUS:-0}"
MAX_RETRIES="${MAX_RETRIES:-20}"
BACKOFF_SECONDS="${BACKOFF_SECONDS:-30}"
LOG_FILE="${LOG_FILE:-run_resilient.log}"

# Portable fallbacks for a minimal Git-for-Windows Bash (discovered running
# this on Windows: no tee/date/sleep/which, only bash itself plus a handful
# of coreutils) -- prefer the real command everywhere it exists (Mac/Linux),
# only fall back to a pure-bash equivalent where it's genuinely missing.
have() { command -v "$1" >/dev/null 2>&1; }

now_utc() {
    if have date; then
        date -u +%Y-%m-%dT%H:%M:%SZ
    else
        printf '%(%Y-%m-%dT%H:%M:%SZ)T\n' -1  # bash 4.2+ builtin; only reached when date is absent
    fi
}

portable_sleep() {
    if have sleep; then
        sleep "$1"
    else
        read -rt "$1" _ < /dev/null 2>/dev/null
        return 0
    fi
}

tee_append() {
    if have tee; then
        tee -a "$1"
    else
        while IFS= read -r line || [ -n "$line" ]; do
            printf '%s\n' "$line"
            printf '%s\n' "$line" >> "$1"
        done
    fi
}

ORIG_ARGS=("$@")
resume_already_present=0
for a in "${ORIG_ARGS[@]}"; do
    if [ "$a" = "--resume" ]; then
        resume_already_present=1
        break
    fi
done
if [ "$resume_already_present" = "1" ]; then
    RESUME_ARGS=("${ORIG_ARGS[@]}")
else
    RESUME_ARGS=("${ORIG_ARGS[@]}" --resume)
fi

log() {
    echo "[$(now_utc)] $*" | tee_append "$LOG_FILE"
}

attempt=0
args=("${ORIG_ARGS[@]}")
while true; do
    attempt=$((attempt + 1))
    log "Attempt $attempt/$MAX_RETRIES: python train.py ${args[*]}"

    "$PYTHON" train.py "${args[@]}" 2>&1 | tee_append "$LOG_FILE"
    code=${PIPESTATUS[0]}

    if [ "$code" -eq 0 ]; then
        if [ "$CONTINUOUS" = "1" ]; then
            log "Chunk finished cleanly (exit 0). CONTINUOUS=1 — starting another chunk with --resume."
            args=("${RESUME_ARGS[@]}")
            continue
        fi
        log "Finished cleanly (exit 0). Done."
        exit 0
    fi

    if [ "$code" -eq 2 ]; then
        log "train.py rejected this invocation (exit 2 — bad args or --expect-params/architecture mismatch). Not retrying; fix the command."
        exit 2
    fi

    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
        log "Hit MAX_RETRIES ($MAX_RETRIES) after exit code $code. Stopping to avoid an infinite crash loop on a paid GPU."
        exit "$code"
    fi

    log "Crashed or was interrupted (exit $code). Retrying in ${BACKOFF_SECONDS}s with --resume."
    args=("${RESUME_ARGS[@]}")
    portable_sleep "$BACKOFF_SECONDS"
done
