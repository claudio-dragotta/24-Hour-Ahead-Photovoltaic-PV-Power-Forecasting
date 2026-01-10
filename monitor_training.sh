#!/bin/bash
# Monitor Multi-Branch Transformer training progress

LOG_FILE="outputs/multi_branch_final.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

echo "🚀 Multi-Branch Transformer Training Monitor"
echo "=============================================="
echo ""

# Check if training is running
if pgrep -f "train_multi_branch.py" > /dev/null; then
    echo "✅ Training is RUNNING"
else
    echo "⚠️  Training process not found (may have finished or crashed)"
fi

echo ""
echo "📊 Latest validation loss improvements:"
echo "----------------------------------------"
grep "val_loss improved" "$LOG_FILE" | tail -10

echo ""
echo "📈 Training epochs completed:"
echo "-----------------------------"
grep -oP "Epoch \d+/\d+" "$LOG_FILE" | tail -1

echo ""
echo "⏱️  Recent log entries (last 5 lines):"
echo "--------------------------------------"
grep -E "(INFO|val_loss|train_loss)" "$LOG_FILE" | tail -5

echo ""
echo "💾 Checkpoint directory:"
ls -lh outputs/multi_branch/final_v1/*.ckpt 2>/dev/null || echo "No checkpoints yet"

echo ""
echo "🔄 To refresh, run: bash monitor_training.sh"
echo "📋 Full log: tail -50 outputs/multi_branch_final.log"
