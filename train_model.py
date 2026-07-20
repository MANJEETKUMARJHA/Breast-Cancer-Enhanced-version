# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump
from utils.feature_names import FEATURE_NAMES

# === CONFIG ===
DATA_PATH = "data/breast-cancer.csv"  # put your CSV here
TARGET_COL = "diagnosis"  # change if your target column is named differently
MODEL_OUTPUT = "model.joblib"
RANDOM_STATE = 42

# Load
df = pd.read_csv(DATA_PATH)

# Normalizing columns (replacing spaces by underscores)
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# If diagnosis is 'M'/'B' convert to 1/0
if df[TARGET_COL].dtype == object:
    df[TARGET_COL] = df[TARGET_COL].map({"M": 1, "B": 0})

# Ensure features exist
missing = [f for f in FEATURE_NAMES if f not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in CSV: {missing}")

X = df[FEATURE_NAMES]
y = df[TARGET_COL]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
])

# Fit
pipeline.fit(X_train, y_train)

# Evaluate (simple)
acc = pipeline.score(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

# Save model pipeline
dump(pipeline, MODEL_OUTPUT)
print(f"Saved pipeline to {MODEL_OUTPUT}")