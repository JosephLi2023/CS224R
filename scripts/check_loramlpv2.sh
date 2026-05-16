#!/usr/bin/env bash
# Show current progress of a loraMLPv2 bake-off run.
#
# Usage:  ./scripts/check_loramlpv2.sh [SEED]
# Example: ./scripts/check_loramlpv2.sh 11
set -e

SEED="${1:-11}"
LOG="/tmp/turnrd_loraMLPv2_seed${SEED}.log"

if [[ ! -f "$LOG" ]]; then
  echo "Log not found: $LOG"
  echo "Has the run been launched? Try: ./scripts/run_loramlpv2.sh $SEED"
  exit 1
fi

echo "=== loraMLPv2 seed=$SEED ==="
LAST_UPDATE=$(stat -f "%Sm" "$LOG" 2>/dev/null || stat -c "%y" "$LOG")
echo "Log: $LOG  (last update: $LAST_UPDATE)"
echo ""

# Orchestrator status (look for any python process running our orchestrator with this seed)
ORCH=$(pgrep -af "run_turnrd_modal.py.*loraMLPv2.*seed $SEED" | awk '{print $1}' | head -1)
if [[ -n "$ORCH" ]]; then
  ETIME=$(ps -p "$ORCH" -o etime= 2>/dev/null | tr -d ' ')
  echo "Orchestrator: ALIVE (PID $ORCH, elapsed $ETIME)"
else
  echo "Orchestrator: NOT RUNNING (completed or stopped)"
fi
echo ""

# Use python for parsing — portable across macOS / Linux, handles Unicode banners.
python3 - "$LOG" << 'PYEOF'
import re
import sys

log_path = sys.argv[1]

re_banner = re.compile(r"Round (\d+): train_(loop|turnrd)")
re_eval   = re.compile(r"Eval done.*pct_success=([\d.]+)")
re_exit   = re.compile(r"Round (\d+): train_(loop|turnrd).*exited \d+ after ([\d.]+)s")

evals  = {}
loop_times = {}
turnrd_times = {}
cur_round = None
cur_phase = None

with open(log_path) as fh:
    for line in fh:
        # Track current round/phase from banners
        m = re_banner.search(line)
        if m and "exited" not in line:
            cur_round = int(m.group(1))
            cur_phase = m.group(2)
            continue
        # Capture eval (only assign to train_loop phase)
        m = re_eval.search(line)
        if m and cur_phase == "loop" and cur_round is not None:
            evals[cur_round] = float(m.group(1))
            continue
        # Capture exit times
        m = re_exit.search(line)
        if m:
            r = int(m.group(1))
            phase = m.group(2)
            t = float(m.group(3))
            if phase == "loop":
                loop_times[r] = t
            else:
                turnrd_times[r] = t

print("Round | Eval  | train_loop time | train_turnrd time")
print("------|-------|-----------------|------------------")
for r in range(5):
    e  = f"{evals[r]:.3f}" if r in evals else "-"
    lt = f"{loop_times[r]:.0f}s" if r in loop_times else "-"
    tt = f"{turnrd_times[r]:.0f}s" if r in turnrd_times else "-"
    print(f"  {r}   | {e:<5} | {lt:<15} | {tt:<17}")
PYEOF
echo ""

# Final status
if grep -q "=== Done\." "$LOG" 2>/dev/null; then
  echo "Status: COMPLETE (all 5 rounds finished)"
elif grep -q "^ERROR" "$LOG" 2>/dev/null; then
  echo "Status: ERROR"
  grep "^ERROR" "$LOG" | tail -2
else
  echo "Status: IN FLIGHT"
fi
echo ""

echo "Note: Some round evals may not stream to local log (Modal --detach quirk)."
echo "If a round's Eval is '-' but train_loop has an exit time, query Modal logs:"
echo "  modal app list"
echo "  modal app logs <ap-XXX> | grep pct_success"
