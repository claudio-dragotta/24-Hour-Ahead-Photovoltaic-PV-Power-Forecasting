"""Ensemble Model Wrapper: treats multiple models as a single unified model.

This module provides a high-level interface to load trained models
and apply ensemble weights automatically, making inference simple.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


class EnsembleModel:
    """Unified ensemble model that combines models automatically.

    This class loads all component models and ensemble weights, providing
    a simple .predict() interface that handles everything internally.

    Example:
        >>> ensemble = EnsembleModel.from_outputs("outputs_ensemble", "outputs_baseline")
        >>> predictions = ensemble.predict(input_data)
    """

    def __init__(self, models: Dict[str, any], weights: Dict[str, float], scalers: Dict[str, any] = None):
        """Initialize ensemble model.

        Args:
            models: Dictionary mapping model names to loaded models.
            weights: Dictionary mapping model names to ensemble weights.
            scalers: Optional dictionary of feature scalers per model.
        """
        self.models = models
        self.weights = weights
        self.scalers = scalers or {}

        # Validate weights sum to 1.0
        total_weight = sum(weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"weights must sum to 1.0, got {total_weight}")

        print(f"[EnsembleModel] initialized with {len(models)} models")
        print(f"[EnsembleModel] ensemble weights: {weights}")

    @classmethod
    def from_outputs(
        cls,
        ensemble_dir: str | Path,
        outputs_dir: str | Path = "outputs_baseline",
    ) -> "EnsembleModel":
        """Load ensemble model from output directories.

        Args:
            ensemble_dir: Path to ensemble outputs (contains ensemble_weights.json).
            outputs_dir: Path to model outputs (lgbm/, cnn/, tft/ subdirs).

        Returns:
            Initialized EnsembleModel instance.

        Example:
            >>> ensemble = EnsembleModel.from_outputs("outputs_ensemble", "outputs_baseline")
        """
        ensemble_dir = Path(ensemble_dir)
        outputs_dir = Path(outputs_dir)

        # Load ensemble weights
        weights_path = ensemble_dir / "ensemble_weights.json"
        if not weights_path.exists():
            raise FileNotFoundError(f"ensemble weights not found: {weights_path}")

        with open(weights_path) as f:
            weights_data = json.load(f)

        model_names = weights_data["model_names"]
        model_weights = dict(zip(model_names, weights_data["weights"]))

        print(f"[EnsembleModel] loading {len(model_names)} models...")

        # Load models
        models = {}
        scalers = {}

        for model_name in model_names:
            # Determine model type (supports both simple and old naming)
            if "LightGBM" in model_name:
                # Load all 24 LightGBM models
                lgbm_models = []
                lgbm_dir = outputs_dir / "lgbm" / "models"
                for h in range(1, 25):
                    model_path = lgbm_dir / f"lgbm_h{h}.joblib"
                    if model_path.exists():
                        lgbm_models.append(joblib.load(model_path))
                    else:
                        raise FileNotFoundError(f"LightGBM model not found: {model_path}")
                models[model_name] = lgbm_models
                print(f"  ✓ {model_name}: 24 LightGBM models")

            elif "CNN-BiLSTM" in model_name or "CNN" in model_name:
                # Load CNN-BiLSTM model and scaler
                model_path = outputs_dir / "cnn" / "model_best.keras"
                scaler_path = outputs_dir / "cnn" / "scalers.joblib"

                if model_path.exists():
                    models[model_name] = tf.keras.models.load_model(model_path)
                    if scaler_path.exists():
                        scalers[model_name] = joblib.load(scaler_path)
                    print(f"  ✓ {model_name}: CNN-BiLSTM + scaler")
                else:
                    raise FileNotFoundError(f"CNN model not found: {model_path}")

            elif "TFT" in model_name:
                # Load TFT model (PyTorch Lightning checkpoint)
                tft_dir = outputs_dir / "tft"
                ckpt_files = list(tft_dir.glob("*.ckpt"))
                if ckpt_files:
                    # TFT requires special loading - store path for now
                    models[model_name] = {"checkpoint_path": ckpt_files[0]}
                    print(f"  ✓ {model_name}: TFT checkpoint found at {ckpt_files[0]}")
                else:
                    print(f"  ⚠ {model_name}: TFT checkpoint not found in {tft_dir}")
                    models[model_name] = None
            else:
                print(f"  ⚠ Unknown model type: {model_name}")
                models[model_name] = None

        return cls(models, model_weights, scalers)

    def predict(
        self, X: pd.DataFrame | np.ndarray, return_individual: bool = False
    ) -> np.ndarray | Dict[str, np.ndarray]:
        """Make predictions using the ensemble.

        Args:
            X: Input features (DataFrame or numpy array).
            return_individual: If True, return individual model predictions too.

        Returns:
            Ensemble predictions (N, 24) array, or dict with individual predictions.

        Example:
            >>> predictions = ensemble.predict(test_data)
            >>> predictions.shape
            (100, 24)  # 100 samples, 24 horizons
        """
        predictions = {}

        for model_name, model in self.models.items():
            if model is None:
                continue  # Skip if model not loaded (e.g., TFT)

            if "LightGBM" in model_name:
                # LightGBM: 24 independent models
                preds_h = []
                for h, lgbm_model in enumerate(model, start=1):
                    pred = lgbm_model.predict(X)
                    preds_h.append(pred)
                predictions[model_name] = np.column_stack(preds_h)  # (N, 24)

            elif "CNN-BiLSTM" in model_name:
                # CNN-BiLSTM: sequence-to-sequence
                scaler = self.scalers[model_name]

                # Scale features
                X_scaled = scaler.transform(X)

                # Reshape for CNN input (assuming windowing already done)
                # predictions[model_name] = model.predict(X_scaled, verbose=0)
                # TODO: Handle windowing for CNN properly
                print(f"  ⚠ {model_name}: CNN prediction not fully implemented yet")
                predictions[model_name] = np.zeros((len(X), 24))

        # Compute weighted ensemble
        ensemble_pred = np.zeros((len(X), 24))
        for model_name, pred in predictions.items():
            weight = self.weights[model_name]
            ensemble_pred += weight * pred

        if return_individual:
            predictions["ensemble"] = ensemble_pred
            return predictions
        else:
            return ensemble_pred

    def save(self, path: str | Path) -> None:
        """Save ensemble configuration (not the models themselves).

        Args:
            path: Path to save ensemble config JSON.
        """
        config = {
            "model_names": list(self.models.keys()),
            "weights": self.weights,
        }

        path = Path(path)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"[EnsembleModel] saved config to {path}")

    def __repr__(self) -> str:
        models_str = ", ".join(self.models.keys())
        return f"EnsembleModel(models=[{models_str}])"
