import unittest

from backend.ml_predictor import predictor
from backend.text_processor import build_ml_input_text
from models.train_model import load_and_preprocess_data


class MLPredictionTests(unittest.TestCase):
    def test_trained_model_returns_actual_probability_and_metrics(self):
        self.assertTrue(predictor.is_loaded)
        details = {
            "job_title": "Backend Engineer",
            "company_name": "Acme Software",
            "job_description": (
                "Design and maintain backend services, build APIs, review code, and collaborate "
                "with product engineers to deliver reliable software."
            ),
            "requirements": "3 years of experience",
            "skills": "Python, SQL, REST APIs",
            "salary": "$90,000 per year",
            "location": "Remote",
            "education": "Bachelor degree",
            "recruiter_email": "recruiter@gmail.com",
            "contact_number": "+1 212 555 0199",
            "application_url": "https://short.example/apply",
            "official_website": "https://acme.example"
        }
        result = predictor.predict(**details)

        model_text = build_ml_input_text(
            job_title=details["job_title"],
            company_name=details["company_name"],
            job_description=details["job_description"],
            requirements=details["requirements"],
            skills=details["skills"],
            salary=details["salary"],
            location=details["location"],
            education=details["education"],
            recruiter_email=details["recruiter_email"],
            contact_number=details["contact_number"],
            application_url=details["application_url"],
            official_website=details["official_website"]
        )
        probabilities = predictor.model.predict_proba(predictor.vectorizer.transform([model_text]))[0]
        class_indices = list(predictor.model.classes_)
        fake_probability = float(probabilities[class_indices.index(1)])
        genuine_probability = float(probabilities[class_indices.index(0)])

        self.assertAlmostEqual(result["fake_probability"], fake_probability, places=4)
        self.assertAlmostEqual(result["genuine_probability"], genuine_probability, places=4)
        self.assertAlmostEqual(sum([result["fake_probability"], result["genuine_probability"]]), 1.0, places=4)
        self.assertTrue(result["suspicious_indicators"])
        self.assertIn(result["prediction_label"], ["Likely Genuine", "Likely Fake"])
        self.assertTrue(result["metrics"])
        self.assertGreater(result["metrics"]["testing_samples"], 0)

    def test_training_data_contains_no_duplicate_combined_text(self):
        frame = load_and_preprocess_data()
        self.assertFalse(frame["combined_text"].duplicated().any())
        self.assertIn("location", frame.columns)
        self.assertIn("salary_range", frame.columns)
        self.assertIn("required_education", frame.columns)


if __name__ == "__main__":
    unittest.main()
