"""Train the talent recommendation model.

This script reads the synthetic dataset produced by ``make_demo_data.py``
and fits a logistic regression model using the feature pipeline defined in
``app/engines/text_engine.py``.  The trained model is saved to
``models/text_model.pkl``.
"""

import os
import pandas as pd

from app.engines.text_engine import TalentRecommender


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_path = os.path.join(project_root, "data", "talent_train.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found at {data_path}. Run scripts/make_demo_data.py first."
        )
    df = pd.read_csv(data_path)
    recommender = TalentRecommender(model_path=os.path.join(project_root, "models", "text_model.pkl"))
    recommender.train(df)
    print("Training completed and model saved.")


if __name__ == "__main__":
    main()