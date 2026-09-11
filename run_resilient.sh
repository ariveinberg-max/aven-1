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
        "$PYTHON" -c 'import sys,time; time.sleep(float(sys.argv[1]))' "$1"
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
RESUME_ARGS=()
skip_init_source=0
checkpoint_dir=checkpoints
read_output=0
for a in "${ORIG_ARGS[@]}"; do
    if [ "$read_output" = "1" ]; then
        checkpoint_dir="$a"
        read_output=0
    fi
    case "$a" in
        --output-dir) read_output=1 ;;
        --output-dir=*) checkpoint_dir="${a#--output-dir=}" ;;
    esac
    if [ "$skip_init_source" = "1" ]; then
        skip_init_source=0
        continue
    fi
    case "$a" in
        --init-from) skip_init_source=1; continue ;;
        --init-from=*) continue ;;
    esac
    RESUME_ARGS+=("$a")
done
if [ "$resume_already_present" != "1" ]; then
    RESUME_ARGS+=(--resume)
fi

choose_restart_args() {
    if [ -f "$checkpoint_dir/latest.pt" ]; then
        args=("${RESUME_ARGS[@]}")
    else
        # Preparation may fail before the first checkpoint exists. A forced
        # --resume would then turn a recoverable failure into an argument error.
        args=("${ORIG_ARGS[@]}")
    fi
}

log() {
    echo "[$(now_utc)] $*" | tee_append "$LOG_FILE"
}

# Pre-flight memory check -- this project has twice lost jobs to macOS
# killing them under system-wide memory pressure (see research/TASKS.md's
# hazard log), and separately hit hours of degraded throughput on Windows
# from the same root cause. This was previously done by hand before every
# run; automating it here so it happens every time, unattended, without
# relying on a human remembering to check `top` first. Warns loudly but
# does not block -- an unattended resilient script can't ask a human
# whether to proceed, so the honest choice is to warn and continue rather
# than silently skip the check or silently refuse to run.
preflight_memory_check() {
    # Pure shell/awk on purpose, not $PYTHON -- this must never touch the
    # same interpreter train.py runs under, since test_resilient.py mocks
    # $PYTHON to record and count every invocation as a training attempt;
    # an extra call here would corrupt that count.
    free_mb=""
    if have vm_stat; then
        page_size=$(vm_stat | head -1 | grep -o '[0-9]\+' | head -1)
        free_pages=$(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); print $3}')
        if [ -n "$page_size" ] && [ -n "$free_pages" ]; then
            free_mb=$(( free_pages * page_size / 1024 / 1024 ))
        fi
    elif have wmic; then
        free_kb=$(wmic OS get FreePhysicalMemory /value 2>/dev/null | tr -d '\r' | grep -o '[0-9]\+')
        [ -n "$free_kb" ] && free_mb=$((free_kb / 1024))
    elif have free; then
        free_mb=$(free -m 2>/dev/null | awk '/^Mem:/ {print $7}')
    fi
    if [ -n "$free_mb" ]; then
        log "Pre-flight: ${free_mb} MB free memory."
        if [ "$free_mb" -lt 800 ] 2>/dev/null; then
            log "WARNING: low free memory (${free_mb} MB) -- this exact condition has caused OOM kills and severe slowdowns on this project before. Proceeding anyway since this is unattended, but this run may be slow or unstable."
        fi
    else
        log "Pre-flight: could not determine free memory on this platform; skipping check."
    fi
}

case "$MAX_RETRIES" in
    ''|*[!0-9]*|0) echo "MAX_RETRIES must be a positive integer." >&2; exit 2 ;;
esac
preflight_memory_check
attempt=0
chunk=0
budget_prefix="resilient-$$-$(now_utc)"
failures=0
stop_requested=0
trap 'stop_requested=1' INT TERM
args=("${ORIG_ARGS[@]}")
while true; do
    if [ "$stop_requested" = "1" ]; then exit 130; fi
    attempt=$((attempt + 1))
    log "Attempt $attempt (consecutive failures $failures/$MAX_RETRIES): python train.py ${args[*]}"

    export AVEN_BUDGET_ID="$budget_prefix-$chunk"
    # caffeinate prevents macOS from sleeping mid-run (no equivalent needed on
    # Windows/Linux -- `have` gates this to only apply where it exists).
    if have caffeinate; then
        caffeinate -i "$PYTHON" train.py "${args[@]}" 2>&1 | tee_append "$LOG_FILE"
    else
        "$PYTHON" train.py "${args[@]}" 2>&1 | tee_append "$LOG_FILE"
    fi
    code=${PIPESTATUS[0]}
    if [ "$stop_requested" = "1" ] || [ "$code" -eq 130 ] || [ "$code" -eq 143 ]; then
        log "Stop requested. Leaving the last saved checkpoint in place; no restart."
        exit 130
    fi

    if [ "$code" -eq 0 ]; then
        failures=0
        if [ "$CONTINUOUS" = "1" ]; then
            log "Chunk finished cleanly (exit 0). CONTINUOUS=1 — starting another chunk with --resume."
            chunk=$((chunk + 1))
            choose_restart_args
            continue
        fi
        log "Finished cleanly (exit 0). Done."
        exit 0
    fi

    if [ "$code" -eq 2 ]; then
        log "train.py rejected this invocation (exit 2 — bad args or --expect-params/architecture mismatch). Not retrying; fix the command."
        exit 2
    fi

    failures=$((failures + 1))
    if [ "$failures" -ge "$MAX_RETRIES" ]; then
        log "Hit MAX_RETRIES ($MAX_RETRIES consecutive failures) after exit code $code. Stopping to avoid an infinite crash loop on a paid GPU."
        exit "$code"
    fi

    log "Crashed or was interrupted (exit $code). Retrying in ${BACKOFF_SECONDS}s from the available checkpoint or original preparation."
    choose_restart_args
    portable_sleep "$BACKOFF_SECONDS"
done
