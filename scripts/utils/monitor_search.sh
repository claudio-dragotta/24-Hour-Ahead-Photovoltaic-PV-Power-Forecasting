#!/bin/bash
# Monitoring script for TFT hyperparameter search

LOG_FILE="/home/claudio/24-Hour-Ahead-Photovoltaic-PV-Power-Forecasting/outputs/logs/random_search_20251208_124350.log"

echo "🔍 TFT Random Search Monitor"
echo "============================="
echo ""

# Check if process is running
if ps aux | grep -q "[t]ft_parallel"; then
    echo "✅ Random search is RUNNING"
    PID=$(ps aux | grep "[t]ft_parallel" | awk '{print $2}')
    echo "   PID: $PID"
    
    # Get runtime
    RUNTIME=$(ps -p $PID -o etime= | tr -d ' ')
    echo "   Runtime: $RUNTIME"
else
    echo "❌ Random search is NOT running"
fi

echo ""
echo "📊 Current Status:"
echo "─────────────────"

# Get latest trial status
tail -100 "$LOG_FILE" 2>/dev/null | grep "Trial status" | tail -1

echo ""
echo "🎯 Latest Trial Configurations:"
echo "────────────────────────────────"

# Show last 3 trials
tail -200 "$LOG_FILE" 2>/dev/null | grep -A 7 "Trial train_tft" | tail -24

echo ""
echo "📈 Progress Summary:"
echo "────────────────────"

# Count trials by status
RUNNING=$(tail -50 "$LOG_FILE" 2>/dev/null | grep "Trial status" | tail -1 | grep -oP '\d+(?= RUNNING)' || echo "0")
TERMINATED=$(tail -50 "$LOG_FILE" 2>/dev/null | grep "Trial status" | tail -1 | grep -oP '\d+(?= TERMINATED)' || echo "0")
PENDING=$(tail -50 "$LOG_FILE" 2>/dev/null | grep "Trial status" | tail -1 | grep -oP '\d+(?= PENDING)' || echo "0")

# Visual progress bar (always show)
TOTAL_TRIALS=150
PROGRESS=$((TERMINATED * 100 / TOTAL_TRIALS))
BAR_WIDTH=50
FILLED=$((PROGRESS * BAR_WIDTH / 100))
EMPTY=$((BAR_WIDTH - FILLED))

echo ""
printf "   Progress: %d/%d (%d%%)\n" $TERMINATED $TOTAL_TRIALS $PROGRESS
printf "   ["
if [ $FILLED -gt 0 ]; then
    printf "\033[32m%${FILLED}s\033[0m" "" | tr ' ' '█'
fi
if [ $EMPTY -gt 0 ]; then
    printf "\033[90m%${EMPTY}s\033[0m" "" | tr ' ' '░'
fi
printf "]\n"

echo ""
echo "   🟢 Running:   $RUNNING trials"
echo "   ✅ Completed: $TERMINATED trials"
echo "   ⏳ Pending:   $PENDING trials"

if [ $TERMINATED -gt 0 ] && [ ! -z "$PID" ]; then
    # Estimate time remaining
    ELAPSED_MIN=$(ps -p $PID -o etimes= 2>/dev/null | awk '{print int($1/60)}')
    if [ ! -z "$ELAPSED_MIN" ] && [ $ELAPSED_MIN -gt 0 ]; then
        AVG_TIME=$((ELAPSED_MIN / TERMINATED))
        REMAINING=$((TOTAL_TRIALS - TERMINATED))
        ETA_MIN=$((REMAINING * AVG_TIME / 3))  # Divide by 3 for concurrent trials
        ETA_HOURS=$((ETA_MIN / 60))
        ETA_MIN_REMAIN=$((ETA_MIN % 60))
        echo ""
        echo "   ⏱️  Avg time:  ${AVG_TIME} min/trial"
        echo "   🎯 ETA:       ~${ETA_HOURS}h ${ETA_MIN_REMAIN}m remaining"
    fi
fi

echo ""
echo "💡 Commands:"
echo "────────────"
echo "   Auto-refresh: watch -n 30 bash $0"
echo "   Watch log: tail -f $LOG_FILE"
echo "   Stop search: pkill -9 -f tft_parallel"
echo ""
