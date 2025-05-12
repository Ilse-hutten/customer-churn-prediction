import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from src.config import DATA_PATH, MODEL_OUTPUT_PATH, RANDOM_STATE
from src.pipeline import build_pipeline

def train_model():
    df = pd.read_csv(DATA_PATH)
    # Drop identifier
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Fix TotalCharges which may contain empty strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Define X and y
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})


    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
    "LogisticRegression": LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestClassifier(
        class_weight="balanced",
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=RANDOM_STATE
    )
}



    param_grids = {
        "LogisticRegression": {
            "classifier__C": [0.1, 1.0, 10.0]
        },
        "RandomForest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        },
        "GradientBoosting": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.01, 0.1],
            "classifier__max_depth": [3, 5],
            'classifier__subsample': [0.7, 0.8, 1.0]
        },
        "XGBoost": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__max_depth": [3, 5],
            'classifier__subsample': [0.7, 1.0],
            'classifier__colsample_bytree': [0.7, 1.0]
        }
    }

    best_score = 0
    best_model = None

    for name, model in models.items():
        pipeline = build_pipeline(model)
        grid = GridSearchCV(pipeline, param_grids[name], cv=3, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"{name} best ROC AUC: {grid.best_score_:.4f}")
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_

    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    print("✅ Best model saved.")
    return best_model, X_test, y_test
