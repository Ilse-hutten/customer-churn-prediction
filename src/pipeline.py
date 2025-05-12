from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from src.preprocess import build_preprocessor

def build_pipeline(model):
    preprocessor = build_preprocessor()
    return Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),  # resample after preprocessing
        ("classifier", model)
    ])
