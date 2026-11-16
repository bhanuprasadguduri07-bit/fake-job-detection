"""
Configuration settings for the Fake Job Recruitment Detection System.
Loads environment variables from .env file when present, with secure fallbacks.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load .env file
load_dotenv(BASE_DIR / ".env")


class Config:
    """Base application configuration."""
    
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production-f83b194a20b")
    
    # Database
    DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "instance" / "database.db"))
    
    # Uploads
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", str(BASE_DIR / "static" / "uploads"))
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16 MB limit
    
    # Machine Learning Model & Vectorizer
    MODEL_DIR = str(BASE_DIR / "models" / "trained_model")
    MODEL_PATH = str(BASE_DIR / "models" / "trained_model" / "fake_job_model.pkl")
    VECTORIZER_PATH = str(BASE_DIR / "models" / "trained_model" / "tfidf_vectorizer.pkl")
    METRICS_PATH = str(BASE_DIR / "models" / "trained_model" / "metrics.json")
    DATASET_PATH = str(BASE_DIR / "dataset" / "fake_job_dataset.csv")
    
    # Search Engine APIs (for official website verification)
    GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "")
    GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "")
    BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY", "")
    
    # SMTP Email Configuration
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "True").lower() in ("true", "1", "yes")
    SMTP_SENDER_EMAIL = os.getenv("SMTP_SENDER_EMAIL", "")
    
    # Automatic Job Monitoring
    MONITORING_INTERVAL_HOURS = int(os.getenv("MONITORING_INTERVAL_HOURS", 24))
    START_MONITORING_SCHEDULER = os.getenv(
        "START_MONITORING_SCHEDULER", "true"
    ).lower() in ("true", "1", "yes")
    
    # Verification & Security
    REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 10))
    MAX_REDIRECTS = 3
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
