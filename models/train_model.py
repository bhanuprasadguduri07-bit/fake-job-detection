"""
Fake Job Recruitment Detection - Machine Learning Training Script
-----------------------------------------------------------------
Dataset: Employment Scam Aegean Dataset (EMSCAD) / Real or Fake Job Postings
Source: University of the Aegean Laboratory of Information and Communication
Systems Security / Public Kaggle Repository
Licence: CC BY-NC-SA 4.0 / Public Research Dataset

This script:
1. Obtains the actual EMSCAD dataset from a verified repository.
2. Cleans missing records and duplicates.
3. Preprocesses text (title, company profile, description, requirements, benefits).
4. Splits data cleanly into training and testing sets (80/20 stratified split).
5. Avoids data leakage by fitting TF-IDF only on training set.
6. Trains a Logistic Regression model with class balancing.
7. Evaluates performance (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
8. Saves the trained model, vectorizer, and metrics using Python's native pickle (NO Joblib).
"""

import os
import sys
import json
import pickle
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "fake_job_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "trained_model")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_job_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

DATASET_URLS = [
    "https://raw.githubusercontent.com/abbylmm/fake_job_posting/main/data/fake_job_postings.csv"
]


def download_dataset():
    """Download the official EMSCAD dataset if not already present."""
    if os.path.exists(DATASET_PATH) and os.path.getsize(DATASET_PATH) > 1000000:
        print(f"Dataset already exists at: {DATASET_PATH} ({os.path.getsize(DATASET_PATH):,} bytes)")
        return True

    print("Downloading Employment Scam Aegean Dataset (EMSCAD)...")
    os.makedirs(DATASET_DIR, exist_ok=True)

    for url in DATASET_URLS:
        try:
            print(f"Attempting download from: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(DATASET_PATH, "wb") as f:
                total_bytes = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    total_bytes += len(chunk)
            print(f"Download completed successfully: {total_bytes:,} bytes saved.")
            return True
        except Exception as e:
            print(f"Failed to download from {url}: {e}")

    return False


def load_and_preprocess_data():
    """Load, inspect, and clean the dataset."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}. Please run download_dataset first.")

    print("\n--- 1. Loading and Inspecting Dataset ---")
    df = pd.read_csv(DATASET_PATH)
    print(f"Raw dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # Target column check
    if "fraudulent" not in df.columns:
        raise ValueError("Missing 'fraudulent' label column in dataset.")

    # Drop records where target is missing or non-binary
    df = df[df["fraudulent"].isin([0, 1])]
    df["fraudulent"] = df["fraudulent"].astype(int)

    fraud_counts = df["fraudulent"].value_counts().to_dict()
    print(f"Class distribution: Genuine (0): {fraud_counts.get(0, 0):,}, Fake (1): {fraud_counts.get(1, 0):,}")

    # Combine all relevant advertised job details for a richer classifier.
    text_fields = [
        "title", "company_profile", "description", "requirements", "benefits",
        "employment_type", "required_experience", "required_education",
        "location", "salary_range", "department", "industry", "function"
    ]
    for col in text_fields:
        if col in df.columns:
            df[col] = (
                df[col].fillna("").astype(str).str.lower()
                .str.replace(r"\s+", " ", regex=True).str.strip()
            )
        else:
            df[col] = ""

    # Remove repeated postings before splitting; conflicting duplicate labels are
    # discarded instead of allowing copies into both train and test partitions.
    initial_rows = len(df)
    duplicate_keys = ["title", "description"]
    conflicting_keys = (
        df.groupby(duplicate_keys, dropna=False)["fraudulent"]
        .nunique()
        .loc[lambda labels: labels > 1]
        .index
    )
    if len(conflicting_keys):
        conflict_index = pd.MultiIndex.from_frame(df[duplicate_keys])
        df = df.loc[~conflict_index.isin(conflicting_keys)].copy()
    df = df.drop_duplicates(subset=duplicate_keys)
    print(f"Removed {initial_rows - len(df)} duplicate/conflicting title and description records.")

    df["combined_text"] = (
        df["title"] + " " +
        df["company_profile"] + " " +
        df["description"] + " " +
        df["requirements"] + " " +
        df["benefits"] + " " +
        df["employment_type"] + " " +
        df["required_experience"] + " " +
        df["required_education"] + " " +
        df["location"] + " " +
        df["salary_range"] + " " +
        df["department"] + " " +
        df["industry"] + " " +
        df["function"]
    ).str.lower().str.strip()

    # Filter out empty texts
    df = df[df["combined_text"].str.len() > 10].copy()
    conflicting_texts = (
        df.groupby("combined_text")["fraudulent"]
        .nunique()
        .loc[lambda labels: labels > 1]
        .index
    )
    df = df[~df["combined_text"].isin(conflicting_texts)].drop_duplicates("combined_text")
    print(f"Usable cleaned records: {len(df):,}")

    return df


def train_and_evaluate():
    """Train the TF-IDF + Logistic Regression model and evaluate metrics."""
    download_dataset()
    df = load_and_preprocess_data()

    X = df["combined_text"]
    y = df["fraudulent"]

    print("\n--- 2. Train-Test Split (Stratified 80/20) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train):,} (Fake: {sum(y_train == 1):,}, Genuine: {sum(y_train == 0):,})")
    print(f"Testing samples:  {len(X_test):,} (Fake: {sum(y_test == 1):,}, Genuine: {sum(y_test == 0):,})")

    print("\n--- 3. TF-IDF Feature Extraction ---")
    # sublinear_tf scales term frequency logarithmically
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )
    # Fit ONLY on training set to avoid data leakage
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,} features")

    print("\n--- 4. Training Logistic Regression Classifier ---")
    # class_weight='balanced' addresses severe class imbalance in real-world recruitment fraud
    model = LogisticRegression(
        class_weight="balanced",
        C=2.0,
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    )
    model.fit(X_train_vec, y_train)
    print("Model training complete.")

    print("\n--- 5. Evaluating Model Performance ---")
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["Genuine", "Fake"], output_dict=True)

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision (Fake class): {precision * 100:.2f}%")
    print(f"Recall (Fake class):    {recall * 100:.2f}%")
    print(f"F1-Score (Fake class):  {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"  True Genuine (TN): {cm[0][0]:,} | False Fake (FP): {cm[0][1]:,}")
    print(f"  False Genuine (FN): {cm[1][0]:,} | True Fake (TP): {cm[1][1]:,}")

    # Most informative suspicious tokens learned by model
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    top_fraud_indices = np.argsort(coefs)[-20:][::-1]
    top_genuine_indices = np.argsort(coefs)[:20]

    top_fraud_features = feature_names[top_fraud_indices].tolist()
    top_genuine_features = feature_names[top_genuine_indices].tolist()
    print("\nTop 10 fraudulent keyword associations:")
    for rank, idx in enumerate(top_fraud_indices[:10], 1):
        print(f"  {rank}. {feature_names[idx]} (weight: {coefs[idx]:.3f})")

    # Save artifacts using native pickle (Strictly NO Joblib!)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nModel saved to: {MODEL_PATH}")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")

    fraud_counts = df["fraudulent"].value_counts().to_dict()
    metrics = {
        "dataset_name": "Employment Scam Aegean Dataset (EMSCAD)",
        "dataset_source": "University of the Aegean / Kaggle Public Datasets",
        "total_records": int(len(df)),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "genuine_count": int(fraud_counts.get(0, 0)),
        "fake_count": int(fraud_counts.get(1, 0)),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "top_fraud_features": top_fraud_features,
        "top_genuine_features": top_genuine_features
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
