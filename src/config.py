import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Telco-Customer-Churn.csv")

MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
RANDOM_STATE = 42
