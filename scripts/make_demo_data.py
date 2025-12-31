"""Generate a synthetic dataset for TrackTalentIR.

This script creates a small synthetic dataset that mimics the
features required for the talent recommendation model.  The dataset
is saved to the ``data/`` directory of the project.  You can run
this script via:

.. code:: bash

   python scripts/make_demo_data.py
"""

import os
import numpy as np
import pandas as pd

from app.engines.text_engine import EVENT_CLASSES


def generate_dataset(n_samples: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame()
    df["age"] = rng.integers(15, 40, size=n_samples)
    df["sex"] = rng.choice(["مرد", "زن"], size=n_samples)
    df["height_cm"] = rng.integers(150, 200, size=n_samples)
    df["weight_kg"] = rng.integers(50, 90, size=n_samples)
    df["sprint_30m_sec"] = rng.uniform(3.5, 6.5, size=n_samples)
    df["run_300m_sec"] = rng.uniform(40.0, 80.0, size=n_samples)
    df["vertical_jump_cm"] = rng.uniform(20.0, 70.0, size=n_samples)
    df["standing_long_jump_cm"] = rng.uniform(120.0, 280.0, size=n_samples)
    df["plank_sec"] = rng.uniform(30.0, 180.0, size=n_samples)
    goals = [
        "افزایش سرعت",
        "افزایش استقامت",
        "پرش بهتر",
        "پرتاب قوی‌تر",
        "سرعت و استقامت",
        "تعادل و انعطاف",
    ]
    df["goal_text"] = rng.choice(goals, size=n_samples)
    # Random labels from the event classes
    df["label_group"] = rng.choice(EVENT_CLASSES, size=n_samples)
    return df


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    df = generate_dataset(300)
    output_path = os.path.join(data_dir, "talent_train.csv")
    df.to_csv(output_path, index=False)
    print(f"Sample dataset written to {output_path}")


if __name__ == "__main__":
    main()