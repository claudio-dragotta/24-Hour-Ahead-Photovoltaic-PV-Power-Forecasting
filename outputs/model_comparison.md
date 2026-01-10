# Model Performance Comparison

## Final Results (Test Set)

| Model | RMSE ↓ | MASE ↓ | Parameters | Notes |
|-------|--------|--------|------------|-------|
| **Multi-Branch Transformer** | **3.31** | 0.46 | 5.0M | ✅ Best RMSE (-10.8% vs TFT) |
| TFT (Baseline) | 3.71 | **0.43** | 2.3M | Strong baseline |

## Key Insights

### Multi-Branch Transformer Advantages
- **10.8% better RMSE** than TFT baseline (3.31 vs 3.71)
- Separate processing branches for PV history, weather history, and weather forecast
- Hierarchical soft attention fusion (adaptive vs fixed weights)
- StandardScaler normalization for all inputs and outputs

### Architecture Comparison
```
Multi-Branch:
- 3 separate transformer branches (PV, Weather Hist, Weather Fcst)
- 2-stage hierarchical fusion with soft attention
- d_model=256, heads=4, layers=2 per branch
- Training: 60 epochs, early stop at val_loss=0.051

TFT:
- Single unified architecture with variable selection
- LSTM encoder/decoder with multi-head attention
- hidden_size=64, heads=4, QuantileLoss
- Training: PyTorch Forecasting baseline configuration
```

## Training Summary

**Multi-Branch Transformer:**
- Training time: ~25 minutes (60 epochs)
- Final val_loss: 0.051
- Best checkpoint: outputs/multi_branch/final_v1/multi-branch-best.ckpt

**TFT Baseline (fixed reference):**
- Configuration: outputs/tft/baseline/config_tft.json
- Best checkpoint: outputs/tft/baseline/tft-best.ckpt

## Conclusion

✅ **Multi-Branch Transformer achieved the best RMSE**, improving by 10.8% over the strong TFT baseline while maintaining competitive MASE performance.
