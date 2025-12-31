"""Talent recommendation engine for TrackTalentIR.

This module contains a simple implementation of a talent recommender for the
running and field events of athletics. It relies on a linear model and a
feature-preprocessing pipeline to map raw user input into predictions about
which event groups (e.g., Sprint, Middle, Endurance, Jumps, Throws, Hurdles)
are most suitable.  The default model is loaded from ``models/text_model.pkl``
if present; otherwise, a trivial fall-back is used which returns a default
recommendation.

To train the model, see ``scripts/train_text.py``.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EVENT_CLASSES = [
    "سرعت",        # Sprint
    "نیمه‌استقامت",  # Middle distance
    "استقامت",      # Endurance
    "پرش‌ها",       # Jumps
    "پرتاب‌ها",      # Throws
    "با مانع",      # Hurdles/Steeple
]


class TalentRecommender:
    """A lightweight recommender for athletic event groups.

    Parameters
    ----------
    model_path : str
        Path to a persisted model file.  If the file does not exist, the
        recommender will fall back to a naive implementation.
    """

    def __init__(self, model_path: str = "models/text_model.pkl"):
        self.model_path = model_path
        self.model: Pipeline | None = None
        # Define columns
        self.cat_cols: List[str] = ["sex"]
        self.num_cols: List[str] = [
            "age",
            "height_cm",
            "weight_kg",
            "sprint_30m_sec",
            "run_300m_sec",
            "vertical_jump_cm",
            "standing_long_jump_cm",
            "plank_sec",
        ]
        self.text_cols: List[str] = ["goal_text"]
        # Attempt to load model
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None

    def train(self, df: pd.DataFrame) -> None:
        """Train the underlying model on the provided DataFrame.

        The DataFrame must contain all columns defined in ``self.num_cols`` and
        ``self.cat_cols``, plus a text column and a ``label_group`` column.
        After training, the model is persisted to ``model_path``.
        """
        X = df[self.num_cols + self.cat_cols + self.text_cols]
        y = df["label_group"]
        # Preprocessing: scale numeric, one-hot encode categorical, vectorize text
        ct = ColumnTransformer(
            [
                ("num", StandardScaler(), self.num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_cols),
                ("text", TfidfVectorizer(), "goal_text"),
            ]
        )
        model = Pipeline(
            [
                ("preprocessor", ct),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )
        model.fit(X, y)
        self.model = model
        # Persist the trained model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)

    def predict(self, input_dict: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Predict the top-3 recommended event groups for a single input sample.

        Parameters
        ----------
        input_dict : dict
            A dictionary mapping feature names to values.  See ``train`` for
            expected keys.

        Returns
        -------
        List[Tuple[str, float]]
            A list of at most three (event, probability) pairs, sorted in
            descending order of probability.  If the model has not been
            trained, a naive recommendation of the first three event classes
            with equal probability is returned.
        """
        # Fall back if model is not trained
        if self.model is None:
            fallback_probs = [1.0 / 3] * 3
            return list(zip(EVENT_CLASSES[:3], fallback_probs))
        # Build DataFrame
        input_df = pd.DataFrame([input_dict])
        try:
            proba = self.model.predict_proba(input_df)[0]
            classes = list(self.model.classes_)
            # Select top 3
            top_indices = np.argsort(proba)[::-1][:3]
            results = [(classes[i], float(proba[i])) for i in top_indices]
            return results
        except Exception:
            # In case of any error during prediction, return fallback
            fallback_probs = [1.0 / 3] * 3
            return list(zip(EVENT_CLASSES[:3], fallback_probs))