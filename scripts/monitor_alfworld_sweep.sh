#!/usr/bin/env bash
# Polls the 3 AlfWorld sweep orchestrators every N minutes. When all 3 die
# (sweeps complete), exits 0 so the calling agent can run the aggregator.
#
# Status report per cycle:
#   - which orchestrators are alive (PID check)
#   - latest few `ep=` lines from each method's log
#   - active Modal app count
#
# Usage: bash scripts/monitor_alfworld_sweep.sh [INTERVAL_MIN]
#   INTERVAL_MIN defaults to 5 (poll every 5 min).
#
# Exit 0 when all 3 sweeps finish (no run_turnrd_modal/run_method_c
# processes alive). Hard-cap of 6 hours so a stuck monitor doesn't loop
# forever.

set -uo pipefail

INTERVAL_MIN=${1:-5}
INTERVAL_S=$((INTERVAL_MIN * 60))
LOG_DIR=${LOG_DIR:-/tmp/alfworld_sweep_logs}
START_TS=$(date +%s)
HARD_CAP_S=$((6 * 3600))

while true; do
    NOW_TS=$(date +%s)
    ELAPSED_M=$(( (NOW_TS - START_TS) / 60 ))

    if [ $((NOW_TS - START_TS)) -gt $HARD_CAP_S ]; then
        echo "[$(date '+%H:%M:%S')] HARD CAP REACHED (6 hr). Exiting monitor."
        exit 124
    fi

    ALIVE_COUNT=$(ps aux | grep -E "run_turnrd_modal.*alfworld|run_method_c_alfworld" | grep -v grep | wc -l | tr -d ' ')
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "[$(date '+%H:%M:%S') | elapsed ${ELAPSED_M}m] alive sweep procs: ${ALIVE_COUNT}"
    echo "════════════════════════════════════════════════════════════════"

    for log in "${LOG_DIR}/method_b_v2.log" "${LOG_DIR}/method_b_lean.log" "${LOG_DIR}/method_c.log"; do
        if [ -f "${log}" ]; then
            method=$(basename "${log}" .log)
            line_count=$(wc -l < "${log}" | tr -d ' ')
            last_ep=$(grep -E "^ep=|═══ Round|Round [0-9]+: train_" "${log}" 2>/dev/null | tail -3 | tr '\n' '|' | sed 's/|/  ★  /g')
            last_eval=$(grep -E "Eval done" "${log}" 2>/dev/null | tail -1)
            echo ""
            echo "── ${method} (${line_count} log lines)"
            if [ -n "${last_ep}" ]; then
                echo "   last events:  ${last_ep}"
            fi
            if [ -n "${last_eval}" ]; then
                echo "   last eval:    ${last_eval}"
            fi
        fi
    done

    if [ "${ALIVE_COUNT}" -eq 0 ]; then
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "[$(date '+%H:%M:%S') | elapsed ${ELAPSED_M}m] ALL SWEEPS COMPLETE"
        echo "════════════════════════════════════════════════════════════════"
        exit 0
    fi

    sleep $INTERVAL_S
done
