"""Combines machine-learning and official-site verification as separate signals."""

import json


def combine_recruitment_assessment(ml_result, official_status):
    """Combine the independent ML and official-site signals without treating absence as proof."""
    failed_statuses = {"Website Check Failed", "UNVERIFIED - WEBSITE CHECK FAILED"}
    verified_statuses = {"Verified on Official Website", "VERIFIED GENUINE"}
    unverified_statuses = {"Unverified", "UNVERIFIED", "Not Found on Official Website"}

    if official_status in failed_statuses:
        return "UNVERIFIED - WEBSITE CHECK FAILED", (
            "The official website could not be reached; this is not evidence that the posting is fake."
        )
    if official_status in unverified_statuses:
        return "UNVERIFIED", "The official website or vacancy could not be confidently verified."
    if official_status == "LIKELY SUSPICIOUS":
        if ml_result and not ml_result.get("is_fake", False):
            return (
                "MANUAL REVIEW REQUIRED",
                "Strong fraud indicators conflict with the ML model's likely-genuine prediction; review both signals."
            )
        return "LIKELY SUSPICIOUS", "Strong fraud indicators were found; review the reported evidence."
    if official_status not in verified_statuses:
        return "UNVERIFIED", "The available ML and website evidence does not support a definitive combined status."
    if ml_result and ml_result.get("prediction_label") == "Unanalyzed":
        return "VERIFIED GENUINE", "A specific published vacancy was confirmed on an official job listing."
    if ml_result and ml_result.get("is_fake"):
        return (
            "MANUAL REVIEW REQUIRED",
            "The official website lists a matching vacancy, but the ML model predicts likely fake; review both signals."
        )
    return (
        "VERIFIED GENUINE",
        "The official listing confirms a specific vacancy and the ML model does not contradict that evidence."
    )


def stored_official_verification_status(verification_result, matching_vacancy=None):
    """Return a safe display status without trusting legacy keyword-match flags."""
    if not verification_result:
        return "UNVERIFIED"
    if matching_vacancy is None:
        raw_details = verification_result.get("matching_vacancy_details")
        if raw_details:
            try:
                matching_vacancy = json.loads(raw_details) if isinstance(raw_details, str) else raw_details
            except (json.JSONDecodeError, TypeError):
                matching_vacancy = None

    has_listing_evidence = bool(
        isinstance(matching_vacancy, dict)
        and matching_vacancy.get("evidence_type") in {
            "schema.org JobPosting",
            "Workday job posting"
        }
        and matching_vacancy.get("job_url")
        and matching_vacancy.get("published_title")
        and matching_vacancy.get("company_name")
    )
    if has_listing_evidence:
        return "VERIFIED GENUINE"
    if (
        verification_result.get("verification_status") == "LIKELY SUSPICIOUS"
        or verification_result.get("official_verification_status") == "LIKELY SUSPICIOUS"
    ):
        return "LIKELY SUSPICIOUS"
    if verification_result.get("official_verification_status") in {
        "Website Check Failed",
        "UNVERIFIED - WEBSITE CHECK FAILED"
    }:
        return "UNVERIFIED - WEBSITE CHECK FAILED"
    return "UNVERIFIED"
