#!/bin/bash
# run_agent.sh — Launch the combinatoric BKG tuning agent
#
# Usage:
#   bash run_agent.sh                   # 5 iterations, llama3
#   bash run_agent.sh --max-iter 10     # 10 iterations
#   bash run_agent.sh --model llama3    # explicit model

AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== SpinQuest Combinatoric BKG Tuning Agent ==="
echo "Working dir : $AGENT_DIR"
echo "Conda env   : root_env"
echo ""

conda run -n root_env python -u "$AGENT_DIR/agent.py" "$@" 2>&1 | tee "$AGENT_DIR/agent_run_$(date +%Y%m%d_%H%M%S).log"
