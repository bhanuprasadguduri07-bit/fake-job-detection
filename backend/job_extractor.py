"""
Job Extractor module for parsing career pages and discovering published vacancies.
Extracts title, location, description, and application links from HTML structures.
"""

import re
import urllib.parse
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from backend.website_verifier import safe_fetch_url, clean_text
from config import Config


def extract_jobs_from_html(html_content: str, base_url: str) -> List[Dict[str, Any]]:
    """
    Parse vacancies from HTML using common careers page patterns, link anchors,
    and schema.org JobPosting markup.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    jobs = []
    seen_titles = set()

    # 1. Look for Schema.org JobPosting items (JSON-LD)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            data = json.loads(script.string)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if item.get("@type") == "JobPosting":
                    title = clean_text(item.get("title", ""))
                    if title and title.lower() not in seen_titles:
                        seen_titles.add(title.lower())
                        loc = item.get("jobLocation", {})
                        loc_str = loc.get("address", {}).get("addressLocality", "") if isinstance(loc, dict) else ""
                        jobs.append({
                            "job_title": title,
                            "job_location": loc_str or "Unspecified",
                            "department": clean_text(item.get("department", "")),
                            "salary": clean_text(item.get("baseSalary", {}).get("value", "")) if isinstance(item.get("baseSalary"), dict) else "",
                            "description": clean_text(item.get("description", ""))[:500],
                            "job_url": item.get("url") or base_url,
                            "external_job_id": item.get("identifier", {}).get("value", "") if isinstance(item.get("identifier"), dict) else ""
                        })
        except Exception:
            continue

    # 2. Look for job links / cards in HTML (common class patterns)
    selectors = [
        "a[href*='job']", "a[href*='career']", "a[href*='position']",
        ".job-title", ".job-card", ".career-card", ".posting-title",
        ".vacancy-item", "li.job", "tr.job-row"
    ]

    for sel in selectors:
        for elem in soup.select(sel):
            title = ""
            url = base_url
            if elem.name == "a":
                title = clean_text(elem.get_text())
                href = elem.get("href", "")
                if href:
                    url = urllib.parse.urljoin(base_url, href)
            else:
                title_elem = elem.find(["h2", "h3", "h4", "a", "strong"])
                if title_elem:
                    title = clean_text(title_elem.get_text())
                link_elem = elem.find("a", href=True)
                if link_elem:
                    url = urllib.parse.urljoin(base_url, link_elem["href"])

            # Filter out generic navigation titles
            if (
                title
                and 4 < len(title) < 90
                and title.lower() not in seen_titles
                and not any(nav in title.lower() for nav in ["home", "about", "contact", "apply now", "view all", "learn more", "privacy policy"])
            ):
                seen_titles.add(title.lower())
                jobs.append({
                    "job_title": title,
                    "job_location": "Office / Remote",
                    "department": "Engineering / Operations",
                    "salary": "Competitive",
                    "description": f"Role: {title}. View official posting details on the careers page.",
                    "job_url": url,
                    "external_job_id": ""
                })

            if len(jobs) >= 20:
                break
        if len(jobs) >= 20:
            break

    # If no specialized selector matched, grab header titles in main/article sections
    if not jobs:
        for heading in soup.find_all(["h2", "h3", "h4"]):
            text = clean_text(heading.get_text())
            if any(k in text.lower() for k in ["engineer", "developer", "manager", "analyst", "designer", "intern", "specialist"]):
                if text.lower() not in seen_titles and len(text) < 80:
                    seen_titles.add(text.lower())
                    jobs.append({
                        "job_title": text,
                        "job_location": "Corporate Location",
                        "department": "General",
                        "salary": "Negotiable",
                        "description": f"Opening for {text}",
                        "job_url": base_url,
                        "external_job_id": ""
                    })

    return jobs


def crawl_career_vacancies(careers_url: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetch careers page and extract all discovered job vacancies.
    Returns (list_of_jobs, error_message).
    """
    html_content, status_code, err = safe_fetch_url(careers_url, timeout=Config.REQUEST_TIMEOUT_SECONDS)
    if err:
        return [], err
    if status_code != 200 or not html_content:
        return [], f"HTTP error {status_code} when loading {careers_url}"

    jobs = extract_jobs_from_html(html_content, careers_url)
    return jobs, None

