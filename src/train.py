"""
Quick Commerce SLA model training and evaluation pipeline.

Loads the prepared training and testing datasets, preprocesses numerical
and categorical features, and trains five candidate classification models:
Logistic Regression, Random Forest, XGBoost, Tuned XGBoost, and LightGBM.

Each model is evaluated using accuracy, precision, recall, F1-score, and
ROC-AUC. MLflow tracks each experiment's parameters, metrics, and model
artifact. The model with the highest ROC-AUC is saved as the best model,
and the evaluation summary is exported to metrics.json.
"""

import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    BEST_MODEL_PATH,
    METRICS_FILE_PATH,
    RANDOM_STATE,
)
from src.logger import logger


def get_preprocessor():
    """Builds a scikit-learn ColumnTransformer for categorical and numerical features."""
    categorical_features = [
        "Weatherconditions",
        "Road_traffic_density",
        "Type_of_order",
        "Type_of_vehicle",
        "Festival",
        "City",
    ]
    numerical_features = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "distance_km",
        "Vehicle_condition",
        "multiple_deliveries",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    return preprocessor #Numeric columns are scaled so features use comparable ranges.

#Categorical columns are encoded into a numerical representation that the model can process.
#The ColumnTransformer applies the appropriate preprocessing to each column type. 

def evaluate_model(model, X_test, y_test):
    """Calculates classification evaluation metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    return metrics


def train_and_evaluate_all_models():
    """Trains 5 candidate models, logs runs to local MLflow, and saves the best model binary."""
    logger.info(f"Loading training data from {TRAIN_DATA_PATH}")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train = train_df.drop(columns=["will_breach_sla"])
    y_train = train_df["will_breach_sla"]
    X_test = test_df.drop(columns=["will_breach_sla"])
    y_test = test_df["will_breach_sla"]

    preprocessor = get_preprocessor()

    models_config = {
        "Logistic_Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=12),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric="logloss"),
        "Tuned_XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1),
    }

    mlflow.set_experiment("Quick_Commerce_SLA_Prediction")
    
    best_roc_auc = -1.0
    best_model_name = ""
    best_pipeline = None
    all_metrics = {}

    for model_name, classifier in models_config.items():
        logger.info(f"--- Training Candidate Model: {model_name} ---")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
# Start an MLflow run to track one model experiment.
# Log model parameters and evaluation metrics for comparison.
# Save the trained model artifact so it can be reused later.
        with mlflow.start_run(run_name=model_name):
            pipeline.fit(X_train, y_train)
            metrics = evaluate_model(pipeline, X_test, y_test)
            
            logger.info(f"Results for {model_name}: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}, Accuracy={metrics['accuracy']:.4f}")

            # Log parameters and metrics to MLflow
            mlflow.log_params({"model_name": model_name, **classifier.get_params()})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            all_metrics[model_name] = metrics

            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_model_name = model_name
                best_pipeline = pipeline

    logger.info(f"🏆 Best Model Selected: {best_model_name} with ROC-AUC score of {best_roc_auc:.4f}")

    # Save Best Model Binary locally
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    logger.info(f"Saved best model pipeline to {BEST_MODEL_PATH}")

    # Export Evaluation Summary to metrics.json
    summary = {
        "best_model": best_model_name,
        "best_roc_auc": best_roc_auc,
        "metrics_by_model": all_metrics,
    }
    with open(METRICS_FILE_PATH, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved execution metrics summary to {METRICS_FILE_PATH}")


if __name__ == "__main__":
    train_and_evaluate_all_models()
