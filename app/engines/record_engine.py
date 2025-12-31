"""Record prediction engine for TrackTalentIR.

This module defines a placeholder implementation for predicting future
performance based on past records.  The current implementation
generates a simple weighted average of past performances with a
random adjustment.  Developers can replace this logic with a proper
regression or classification model trained on real competition data.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np


class RecordPredictor:
    """A trivial predictor for athletic performance.

    The ``predict`` method expects a dictionary containing the keys
    ``pr_time`` (personal best) and ``last1``, ``last2``, ``last3`` for the
    most recent competition results.  It returns a prediction of the next
    performance and an optional note.
    """

    def predict(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            pr_time = float(input_dict.get("pr_time", 0))
            last1 = float(input_dict.get("last1", 0))
            last2 = float(input_dict.get("last2", 0))
            last3 = float(input_dict.get("last3", 0))
            # Weighted average: more weight for recent performances
            weighted_avg = (last1 * 0.5 + last2 * 0.3 + last3 * 0.2)
            # Random adjustment within ±5%
            adjustment = weighted_avg * np.random.uniform(-0.05, 0.05)
            predicted = weighted_avg + adjustment
            # Ensure predicted does not beat personal record unrealistically
            predicted_time = max(predicted, pr_time * 0.98)
            notes = "پیش‌بینی بر اساس میانگین وزنی رکوردهای اخیر و کمی تغییر تصادفی است."
            return {"predicted_time": predicted_time, "notes": notes}
        except Exception:
            return {"predicted_time": None, "notes": "خطا در ورودی"}