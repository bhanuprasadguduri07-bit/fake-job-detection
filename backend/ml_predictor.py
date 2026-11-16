"""
Machine Learning Predictor for Fake Job Recruitment Detection.
Loads the trained model and vectorizer, generates predictions, confidence scores,
suspicious indicators, and human-readable explanations.
"""

import os
import json
import pickle
from typing import Dict, Any, Optional
from config import Config
from backend.text_processor import (
    build_ml_input_text,
    detect_suspicious_indicators
)


class MLPredictor:
    """Predicts whether a job posting is likely fake or genuine using TF-IDF and Logistic Regression."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLPredictor, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.vectorizer = None
            cls._instance.metrics = {}
            cls._instance.is_loaded = False
            cls._instance.load_model()
        return cls._instance

    def load_model(self) -> bool:
        """Load the trained model, vectorizer, and evaluation metrics using native pickle."""
        if not os.path.exists(Config.MODEL_PATH) or not os.path.exists(Config.VECTORIZER_PATH):
            print(f"Warning: Model files not found at {Config.MODEL_PATH}")
            self.is_loaded = False
            return False

        try:
            with open(Config.MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)

            with open(Config.VECTORIZER_PATH, "rb") as f:
                self.vectorizer = pickle.load(f)

            if (
                not hasattr(self.model, "predict_proba")
                or not hasattr(self.vectorizer, "transform")
                or set(getattr(self.model, "classes_", [])) != {0, 1}
            ):
                raise ValueError("The loaded model must expose probabilities for both labels 0 and 1.")

            if os.path.exists(Config.METRICS_PATH):
                with open(Config.METRICS_PATH, "r") as f:
                    self.metrics = json.load(f)
            else:
                self.metrics = {}

            self.is_loaded = True
            print("ML model and vectorizer successfully loaded.")
            return True
        except Exception as e:
            print(f"Error loading ML model: {e}")
            self.is_loaded = False
            return False

    def predict(
        self,
        job_title: str,
        company_name: str,
        job_description: str,
        requirements: Optional[str] = "",
        skills: Optional[str] = "",
        salary: Optional[str] = "",
        location: Optional[str] = "",
        recruiter_email: Optional[str] = "",
        application_url: Optional[str] = "",
        education: Optional[str] = "",
        contact_number: Optional[str] = "",
        official_website: Optional[str] = ""
    ) -> Dict[str, Any]:
        """
        Analyze a job advertisement and return prediction, confidence, indicators, and explanation.
        """
        combined_text = build_ml_input_text(
            job_title=job_title,
            company_name=company_name,
            job_description=job_description,
            requirements=requirements,
            skills=skills,
            salary=salary,
            location=location,
            education=education,
            recruiter_email=recruiter_email,
            contact_number=contact_number,
            application_url=application_url,
            official_website=official_website
        )

        word_count = len(combined_text.split())
        insufficient_info_warning = None
        if word_count < 15:
            insufficient_info_warning = (
                f"Notice: Only {word_count} words were provided. Job descriptions with very little text "
                "give limited linguistic context for machine learning analysis. Results should be treated with caution."
            )

        # Detect heuristic/semantic suspicious indicators
        full_text_for_indicators = (
            f"{job_title} {company_name} {job_description} {requirements} {skills} "
            f"{education} {location} {contact_number}"
        )
        suspicious_indicators = detect_suspicious_indicators(
            text=full_text_for_indicators,
            recruiter_email=recruiter_email,
            company_name=company_name,
            application_url=application_url,
            salary=salary
        )

        if not self.is_loaded:
            loaded = self.load_model()
            if not loaded:
                return {
                    "prediction_label": "Unanalyzed",
                    "is_fake": False,
                    "confidence_score": 0.0,
                    "fake_probability": 0.0,
                    "suspicious_indicators": suspicious_indicators,
                    "explanation": "ML model is not yet trained or loaded. Please train the model via 'python models/train_model.py'.",
                    "insufficient_info_warning": insufficient_info_warning,
                    "metrics": self.metrics
                }

        # Vectorize text
        vec = self.vectorizer.transform([combined_text])
        probabilities = self.model.predict_proba(vec)[0]
        class_indices = list(self.model.classes_)
        fake_prob = float(probabilities[class_indices.index(1)])
        genuine_prob = float(probabilities[class_indices.index(0)])

        is_fake = fake_prob >= 0.50
        prediction_label = "Likely Fake" if is_fake else "Likely Genuine"
        confidence_score = round(fake_prob if is_fake else genuine_prob, 4)

        explanation = (
            f"The trained text classifier assigned {fake_prob:.1%} fake probability and "
            f"{genuine_prob:.1%} genuine probability."
        )

        return {
            "prediction_label": prediction_label,
            "is_fake": bool(is_fake),
            "confidence_score": confidence_score,
            "fake_probability": round(fake_prob, 4),
            "genuine_probability": round(genuine_prob, 4),
            "suspicious_indicators": suspicious_indicators,
            "explanation": explanation,
            "insufficient_info_warning": insufficient_info_warning,
            "metrics": self.metrics
        }


# Global singleton instance
predictor = MLPredictor()
