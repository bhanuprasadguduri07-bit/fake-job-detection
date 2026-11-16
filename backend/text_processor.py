"""
Text processing and feature analysis for Fake Job Recruitment Detection.
Provides text cleaning, entity extraction, and heuristic suspicious indicator detection.
"""

import re
import html
from typing import List, Dict, Any, Optional

# Known free/disposable email domains often abused in recruiter impersonation
FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "protonmail.com", "proton.me", "mail.com", "zoho.com", "yandex.com",
    "gmx.com", "tempmail.com", "guerrillamail.com", "10minutemail.com",
    "trashmail.com", "sharklasers.com", "getairmail.com"
}

# Suspicious phrases commonly seen in fraudulent recruitment advertisements
SUSPICIOUS_PATTERNS = [
    (
        r"(wire\s+transfer|western\s+union|moneygram|zelle|cash\s*app|venmo|bitcoin|cryptocurrency|gift\s*card)",
        "Demands payment or money transfer via untraceable methods (crypto, wire, gift cards)."
    ),
    (
        r"(registration\s+fee|application\s+fee|training\s+fee|processing\s+fee|background\s+check\s+fee|security\s+deposit|pay\s+for\s+training)",
        "Requires job seekers to pay an upfront fee for application, background checks, or equipment."
    ),
    (
        r"(send\s+(you\s+)?a\s+check\s+to\s+(buy|purchase)|cash\s+(this\s+)?check|reimburse\s+us|overpayment)",
        "Classic fake check scam: offers to send a check to purchase home office equipment."
    ),
    (
        r"(telegram\s+app|telegram\s+username|contact\s+.*@telegram|whatsapp\s+interview|interview\s+via\s+whatsapp|telegram\s+interview)",
        "Requests informal interview via encrypted chat (Telegram or WhatsApp) instead of corporate platforms."
    ),
    (
        r"(no\s+interview\s+needed|immediate\s+offer|guaranteed\s+job|no\s+experience\s+needed.*\$[5-9]\d{2,}|earn\s+\$[1-9]\d{3,}\s*(per|/)\s*week\s+part\s*time)",
        "Unrealistic promises: immediate job offer without interview or excessive pay for unskilled work."
    ),
    (
        r"(send\s+(your\s+)?(ssn|social\s+security|credit\s+card|bank\s+account\s+details|online\s+banking\s+password))",
        "Requests sensitive personal or banking credentials prior to formal recruitment process."
    ),
    (
        r"(package\s+forwarding|reshipment\s+agent|mystery\s+shopper\s+evaluation|envelope\s+stuffing)",
        "Known high-risk fraudulent job category (package reshipping or check cashing mule)."
    )
]


def clean_text(text: Optional[str]) -> str:
    """Normalize and clean text for processing and feature extraction."""
    if not text:
        return ""
    # Decode HTML entities
    text = html.unescape(text)
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text."""
    if not text:
        return []
    pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    return list(dict.fromkeys(re.findall(pattern, text)))


def extract_phone_numbers(text: str) -> List[str]:
    """Extract standard telephone numbers from text."""
    if not text:
        return []
    pattern = r"(\+?\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,5}"
    matches = re.finditer(pattern, text)
    phones = []
    for m in matches:
        phone = m.group(0).strip()
        # Filter out short or likely date patterns
        digits = re.sub(r"\D", "", phone)
        if 7 <= len(digits) <= 15:
            phones.append(phone)
    return list(dict.fromkeys(phones))


def extract_urls(text: str) -> List[str]:
    """Extract web URLs from text."""
    if not text:
        return []
    pattern = r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
    return list(dict.fromkeys(re.findall(pattern, text)))


def check_recruiter_email(email: Optional[str], company_name: Optional[str]) -> Optional[str]:
    """Check if the recruiter's email domain is suspicious for the claimed company."""
    if not email or "@" not in email:
        return None
    domain = email.split("@")[-1].lower()
    if domain in FREE_EMAIL_DOMAINS:
        company_clean = (company_name or "").strip()
        if company_clean and len(company_clean) > 2:
            return (
                f"Recruiter uses a free public email provider (@{domain}) while claiming to represent "
                f"'{company_clean}'. Legitimate corporate recruiters typically use official corporate domain emails."
            )
        return f"Recruiter contact is a free public email service (@{domain}) rather than an organizational domain."
    return None


def detect_suspicious_indicators(
    text: str,
    recruiter_email: Optional[str] = None,
    company_name: Optional[str] = None,
    application_url: Optional[str] = None,
    salary: Optional[str] = None
) -> List[str]:
    """Identify concrete suspicious flags and fraud indicators from all available details."""
    indicators = []
    cleaned_full_text = clean_text(text).lower()

    # 1. Pattern matching on job text
    for regex, description in SUSPICIOUS_PATTERNS:
        if re.search(regex, cleaned_full_text, re.IGNORECASE):
            if description not in indicators:
                indicators.append(description)

    # 2. Check recruiter email
    if recruiter_email:
        email_warning = check_recruiter_email(recruiter_email, company_name)
        if email_warning:
            indicators.append(email_warning)
    else:
        # Check if email is embedded in description
        embedded_emails = extract_emails(cleaned_full_text)
        for emb in embedded_emails:
            emb_warn = check_recruiter_email(emb, company_name)
            if emb_warn and emb_warn not in indicators:
                indicators.append(emb_warn)
                break

    # 3. Check application URL
    if application_url:
        app_url_lower = application_url.lower()
        if any(susp in app_url_lower for susp in ["bit.ly", "tinyurl.com", "t.me", "wa.me", "forms.gle"]):
            indicators.append(
                f"Application uses a URL shortener or unbranded form service ({application_url}) instead of a corporate careers portal."
            )

    # 4. Check salary reasonableness
    if salary:
        salary_clean = salary.lower()
        if re.search(r"(\$|usd|inr|eur)\s*([5-9]\d{3,}|[1-9]\d{4,})\s*(per|/)?\s*(week|day)", salary_clean):
            indicators.append(
                f"Advertised salary ({salary}) appears disproportionately high for standard employment, a common hallmark of recruitment scams."
            )

    return indicators


def build_ml_input_text(
    job_title: str,
    company_name: str,
    job_description: str,
    requirements: Optional[str] = "",
    skills: Optional[str] = "",
    salary: Optional[str] = "",
    location: Optional[str] = "",
    education: Optional[str] = "",
    recruiter_email: Optional[str] = "",
    contact_number: Optional[str] = "",
    application_url: Optional[str] = "",
    official_website: Optional[str] = ""
) -> str:
    """Compose the combined text string expected by the TF-IDF vectorizer."""
    parts = [
        job_title or "",
        company_name or "",
        job_description or "",
        requirements or "",
        skills or "",
        salary or "",
        location or "",
        education or "",
        recruiter_email or "",
        contact_number or "",
        application_url or "",
        official_website or ""
    ]
    combined = " ".join([clean_text(p) for p in parts if p])
    return combined.lower().strip()
