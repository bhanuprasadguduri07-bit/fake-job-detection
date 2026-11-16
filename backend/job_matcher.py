"""
Job Matching System.
Calculates relevance scores between user preferences and job vacancies.
Maintains strict separation between vacancy match score and ML fraud prediction.
"""

import re
from typing import Dict, Any, List, Optional
from backend.text_processor import clean_text


def calculate_token_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two token sets."""
    tokens1 = set(re.findall(r"\w+", (text1 or "").lower()))
    tokens2 = set(re.findall(r"\w+", (text2 or "").lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


def calculate_skills_match(required_skills: str, job_text: str) -> float:
    """
    Check what fraction of the user's required skills appear in the job posting text.
    Returns score between 0.0 and 1.0.
    """
    if not required_skills or not required_skills.strip():
        return 1.0  # No skill filter requested = 100% skill satisfaction

    # Parse comma or whitespace separated skills
    skills_list = [s.strip().lower() for s in re.split(r"[,;|/\n]+", required_skills) if s.strip()]
    if not skills_list:
        return 1.0

    job_text_lower = (job_text or "").lower()
    matched_count = 0
    for skill in skills_list:
        # Match whole word or exact token
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, job_text_lower):
            matched_count += 1

    return matched_count / len(skills_list)


def calculate_job_match(
    user_req: Dict[str, Any],
    job_title: str,
    company_name: str,
    job_description: str,
    job_location: Optional[str] = "",
    salary: Optional[str] = "",
    experience: Optional[str] = "",
    job_skills: Optional[str] = ""
) -> Dict[str, Any]:
    """
    Compute detailed multi-criteria job match score against user profile requirements.
    Weights:
      - Job Title Match: 30%
      - Skills Match: 35%
      - Location Match: 15%
      - Experience Match: 10%
      - Company / Salary Match: 10%
    """
    pref_title = user_req.get("preferred_title", "")
    pref_company = user_req.get("preferred_company", "")
    req_skills = user_req.get("required_skills", "")
    pref_loc = user_req.get("preferred_location", "")
    min_sal = float(user_req.get("minimum_salary") or 0.0)
    pref_exp = user_req.get("experience_level", "")

    full_job_content = f"{job_title} {job_description} {job_skills} {experience}"

    # 1. Title Similarity (30%)
    if pref_title:
        title_sim = calculate_token_similarity(pref_title, job_title)
        # Bonus if full keyword appears
        if pref_title.lower() in job_title.lower():
            title_sim = max(title_sim, 0.90)
    else:
        title_sim = 1.0

    # 2. Skills Match (35%)
    skills_sim = calculate_skills_match(req_skills, full_job_content)

    # 3. Location Match (15%)
    if pref_loc and pref_loc.strip():
        loc_pref_clean = pref_loc.strip().lower()
        job_loc_clean = (job_location or "").lower()
        if (
            loc_pref_clean in job_loc_clean
            or "remote" in job_loc_clean
            or "anywhere" in job_loc_clean
            or "hybrid" in job_loc_clean
        ):
            loc_sim = 1.0
        elif loc_pref_clean in full_job_content.lower():
            loc_sim = 0.8
        else:
            loc_sim = 0.2
    else:
        loc_sim = 1.0

    # 4. Experience Match (10%)
    if pref_exp and pref_exp.strip():
        exp_pref_clean = pref_exp.strip().lower()
        exp_job_clean = (experience or "").lower()
        if exp_pref_clean in exp_job_clean or exp_pref_clean in full_job_content.lower():
            exp_sim = 1.0
        else:
            exp_sim = 0.5
    else:
        exp_sim = 1.0

    # 5. Company / Salary Preference (10%)
    comp_sal_sim = 0.5
    if pref_company and pref_company.strip():
        if pref_company.lower() in (company_name or "").lower():
            comp_sal_sim = 1.0
        else:
            comp_sal_sim = 0.3
    else:
        comp_sal_sim = 0.8

    # Weighted calculation
    total_score = (
        (title_sim * 30.0) +
        (skills_sim * 35.0) +
        (loc_sim * 15.0) +
        (exp_sim * 10.0) +
        (comp_sal_sim * 10.0)
    )
    total_score = round(min(100.0, max(0.0, total_score)), 1)
    qualifies = total_score >= 70.0

    breakdown = {
        "title_match_percentage": round(title_sim * 100, 1),
        "skills_match_percentage": round(skills_sim * 100, 1),
        "location_match_percentage": round(loc_sim * 100, 1),
        "experience_match_percentage": round(exp_sim * 100, 1),
        "company_match_percentage": round(comp_sal_sim * 100, 1)
    }

    return {
        "match_score": total_score,
        "qualifies_for_alert": qualifies,
        "title_match_score": round(title_sim * 100, 1),
        "skills_match_score": round(skills_sim * 100, 1),
        "location_match_score": round(loc_sim * 100, 1),
        "experience_match_score": round(exp_sim * 100, 1),
        "salary_match_score": round(comp_sal_sim * 100, 1),
        "breakdown": breakdown,
        "explanation": (
            f"Overall match score is {total_score}%. "
            f"(Title: {breakdown['title_match_percentage']}%, Skills: {breakdown['skills_match_percentage']}%, "
            f"Location: {breakdown['location_match_percentage']}%). "
            + ("Qualifies for high-relevance alert (>= 70%)." if qualifies else "Below 70% notification threshold.")
        )
    }

