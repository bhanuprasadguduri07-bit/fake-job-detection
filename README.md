# Fake Job Detection

Fake Job Detection is a Flask web application that analyzes job postings for
potential recruitment fraud. It combines a trained machine-learning model,
poster OCR, and company/job website verification to provide an assessment.

## Features

- Analyze job descriptions with a trained TF-IDF and Logistic Regression model.
- Upload recruitment posters and extract text using Tesseract OCR.
- Verify company websites and look for matching vacancies.
- Review analysis results and history in the web interface.
- Track monitored career websites and job matches.
- Store application data in a local SQLite database.

## Technologies

- Python and Flask
- HTML, CSS, JavaScript, Bootstrap, and Jinja templates
- scikit-learn, pandas, and NumPy
- Tesseract OCR via `pytesseract`
- SQLite
- Requests and Beautiful Soup for website checks
- APScheduler for periodic monitoring

## Project structure

```text
.
├── app.py                       # Flask application and routes
├── config.py                    # Environment-based application settings
├── database.py                  # SQLite initialization and access
├── backend/                     # OCR, prediction, verification, and services
├── templates/                   # Flask/Jinja HTML templates
├── static/
│   ├── css/                     # Stylesheets
│   ├── images/                  # UI illustrations
│   ├── js/                      # Browser-side JavaScript
│   └── uploads/                 # Runtime poster uploads (not tracked)
├── models/
│   ├── train_model.py           # Dataset download and model training
│   └── trained_model/           # Runtime model, vectorizer, and metrics
├── dataset/                     # Training data downloaded when needed
├── instance/                    # Runtime SQLite database (not tracked)
├── tests/                       # Automated tests
├── .env.example                 # Example environment configuration
├── .gitignore
└── requirements.txt
```

## Requirements

- Python 3.10 or later
- Tesseract OCR installed separately for poster text extraction
- Git, if you want to clone or contribute to this repository

Website-search API credentials and SMTP settings are optional. Without search
API credentials, company verification may use the application's other available
verification behavior; email notifications require SMTP configuration.

## Installation

In PowerShell, from the project directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Install Tesseract OCR for your operating system. On Windows, the default
installation path used by the OCR module is
`C:\Program Files\Tesseract-OCR\tesseract.exe`. If Tesseract is installed
elsewhere, set `TESSERACT_PATH` in `.env`.

Edit `.env` and set a new, random `SECRET_KEY` before using accounts. Add
optional Google or Bing search credentials and SMTP settings there if needed.
Never commit `.env` or put real credentials in `.env.example`.

## Run the Flask application

```powershell
python app.py
```

Open the local address printed by Flask (typically `http://127.0.0.1:5000/`).
The app initializes its SQLite database under `instance/` when it starts.

## How to use

1. Open the home page and choose text analysis, poster upload, or company
   verification from the navigation.
2. Enter job details or upload a recruitment poster. For poster uploads, review
   and edit the OCR-extracted information before analysis.
3. Submit the information to view the machine-learning prediction and any
   available company/job verification results.
4. Sign in to view account features such as analysis history, preferences, and
   website monitoring.

The included trained model and vectorizer are required for prediction and are
kept in `models/trained_model/`. The training dataset is not included: it is
not needed to run the application, and `models/train_model.py` downloads it
when training is requested. The dataset source has its own licensing and
attribution requirements; review those terms before using or redistributing it.

## Tests

Run the test suite from the project directory:

```powershell
python -m unittest discover -s tests -v
```

## Security and local data

The SQLite database and uploaded posters are runtime data and are excluded from
version control because they may contain account details, job submissions, or
personal information. Configure a strong `SECRET_KEY` and keep credentials in
your local `.env` file. Do not use the development fallback secret for a
deployed application.
