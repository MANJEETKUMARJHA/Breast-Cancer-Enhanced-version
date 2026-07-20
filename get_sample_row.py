import pandas as pd
import numpy as np
from joblib import load
from utils.feature_names import FEATURE_NAMES
import os

df = pd.read_csv("data/breast-cancer.csv")
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

print("\nAnalyzing dataset to find the best test cases...")

model_path = "model.joblib"
if os.path.exists(model_path):
    model = load(model_path)
    X = df[FEATURE_NAMES]
    
    # Get probabilities for being Cancerous (Class 1)
    probs = model.predict_proba(X)[:, 1]
    
    # 1. Clear Benign (Lowest probability of being cancerous)
    idx_benign = np.argmin(probs)
    
    # 2. Clear Malignant (Highest probability of being cancerous)
    idx_malignant = np.argmax(probs)
    
    # 3. Moderate / Borderline (Probability closest to 50%)
    idx_moderate = np.argmin(np.abs(probs - 0.5))
    
    indices = [idx_benign, idx_moderate, idx_malignant]
    labels = [
        "Clearly NOT Cancerous (Benign - High Confidence)", 
        "Moderate / Borderline (Low Confidence)", 
        "Clearly Cancerous (Malignant - High Confidence)"
    ]
else:
    # Fallback if model isn't found
    indices = df.sample(3).index.tolist()
    labels = ["Random 1", "Random 2", "Random 3"]

print("==============================================================")
for i, idx in enumerate(indices):
    row = df.iloc[idx]
    actual_diagnosis = row.get("diagnosis", "Unknown")
    sample = row[FEATURE_NAMES]
    
    print(f"\n--- {labels[i]} ---")
    print(f"Actual Dataset Label: {actual_diagnosis}")
    print("Copy-paste this row:")
    print(",".join(str(v) for v in sample.tolist()))

print("\n==============================================================")
