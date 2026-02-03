import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/Placement_Prediction_data.csv"
MODEL_PATH = "app/ml/models/placement_model.pkl"
TARGET_COL = "PlacementStatus"

def main():
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    # Convert target to 0/1
    df[TARGET_COL] = df[TARGET_COL].map({"Placed": 1, "NotPlaced": 0})

    if df[TARGET_COL].isna().any():
        raise ValueError("PlacementStatus has invalid values. Expected only Placed/NotPlaced.")

    y = df[TARGET_COL]

    X = df.drop(columns=[TARGET_COL])

    if "StudentId" in X.columns:
        X = X.drop(columns=["StudentId"])

    categorical_cols = [
    "Internship",
    "Hackathon"
    ]

    numeric_cols = [
    "CGPA",
    "Projects",
    "Workshops/Certifications",
    "Skills",
    "Communication Skill Rating",
    "12th Percentage",
    "10th Percentage",
    "backlogs"
    ]


    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Placement model trained successfully")
    print("Accuracy:", round(acc, 4))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    joblib.dump(pipeline, MODEL_PATH)
    print("Saved model to:", MODEL_PATH)

if __name__ == "__main__":
    main()
