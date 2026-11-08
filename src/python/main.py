"""Titanic survival prediction using scikit-learn logistic regression.

Data files must be downloaded from the Kaggle Titanic competition.
They are kept out of Git using the repository .gitignore.
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path(
    os.environ.get(
        "TITANIC_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")
    )
)


def prepare_features(df):
    """Create an interpretable feature using family relationships."""
    result = df.copy()
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    return result


def build_model():
    numeric_features = ["Pclass", "Age", "Fare", "FamilySize"]
    categorical_features = ["Sex", "Embarked"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    return model, numeric_features + categorical_features


def main():
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    print("=== Titanic Survival Prediction (Python) ===")
    print(f"Looking for data in: {DATA_DIR}")
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "Missing train.csv or test.csv. Download both from the Kaggle "
            "Titanic competition, place them in src/data/, and follow the "
            "README Docker run command."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"\n[1] Loaded train.csv: {train.shape[0]} rows, {train.shape[1]} columns")
    print(f"    Loaded test.csv:  {test.shape[0]} rows, {test.shape[1]} columns")
    print("    Survival labels in train.csv:")
    print(train["Survived"].value_counts().sort_index().to_string())
    print("\n[2] Missing values in the original training features:")
    model, features = build_model()
    print(train[[c for c in features if c != "FamilySize"]].isna().sum().to_string())

    train = prepare_features(train)
    test = prepare_features(test)
    print("\n[3] Created FamilySize = SibSp + Parch + 1 in both datasets.")
    print("    Selected features:", ", ".join(features))
    print("    Numeric missing values -> training-set median; categorical -> most frequent.")
    print("    Numeric variables are standardized; categories are one-hot encoded.")

    X = train[features]
    y = train["Survived"]
    X_build, X_holdout, y_build, y_holdout = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_build, y_build)
    train_accuracy = accuracy_score(y_build, model.predict(X_build))
    holdout_accuracy = accuracy_score(y_holdout, model.predict(X_holdout))
    print("\n[4] Logistic regression evaluated with an 80/20 stratified split:")
    print(f"    Training rows: {len(X_build)} | Held-out rows: {len(X_holdout)}")
    print(f"    Training accuracy: {train_accuracy:.4f} ({train_accuracy:.2%})")
    print(f"    Held-out accuracy: {holdout_accuracy:.4f} ({holdout_accuracy:.2%})")

    # Refit on all labeled rows before generating final Kaggle test predictions.
    model.fit(X, y)
    full_train_accuracy = accuracy_score(y, model.predict(X))
    test_predictions = model.predict(test[features])
    print("\n[5] Refit model on all labeled train.csv rows.")
    print(f"    Full training-data accuracy (in-sample): {full_train_accuracy:.4f}")
    print(f"    Generated {len(test_predictions)} predictions for Kaggle test.csv.")
    print(f"    Predicted survivors: {int(test_predictions.sum())}")
    print("    First 10 predictions (PassengerId, Survived):")
    preview = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": test_predictions}
    )
    print(preview.head(10).to_string(index=False))
    print("\n[6] Kaggle test.csv does not contain true Survived labels.")
    print("    Its accuracy cannot be computed from these two files.")
    print("    The held-out accuracy above is the unbiased evaluation estimate.")
    print("\nPython pipeline completed successfully.")


if __name__ == "__main__":
    main()