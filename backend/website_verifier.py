"""
Official Company Website Verification Module.
Implements SSRF protection, safe web crawling, domain matching,
ATS platform verification, and evidence aggregation.
"""

import re
import json
import socket
import ipaddress
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from config import Config
from backend.search_engine import SearchEngineService
from backend.text_processor import clean_text

# Recognized authorized Applicant Tracking Systems (ATS) and recruitment platforms
AUTHORIZED_RECRUITMENT_PLATFORMS = {
    "greenhouse.io": "Greenhouse ATS",
    "boards.greenhouse.io": "Greenhouse ATS",
    "lever.co": "Lever ATS",
    "jobs.lever.co": "Lever ATS",
    "myworkdayjobs.com": "Workday Recruiting",
    "smartrecruiters.com": "SmartRecruiters",
    "taleo.net": "Oracle Taleo",
    "icims.com": "iCIMS Talent Cloud",
    "ashbyhq.com": "Ashby Recruiting",
    "jobvite.com": "Jobvite",
    "workable.com": "Workable",
    "bamboohr.com": "BambooHR",
    "recruitee.com": "Recruitee",
    "linkedin.com": "LinkedIn Official Jobs",
    "indeed.com": "Indeed Verified Posting",
}

# Blocked IP networks for SSRF protection
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.88.99.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("255.255.255.255/32"),
    # IPv6 blocked ranges
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("ff00::/8")
]

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {80, 443, 8080}


def is_safe_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Strict SSRF validation:
    1. Validates scheme (HTTP/HTTPS only).
    2. Validates port (80, 443, 8080).
    3. Resolves DNS and blocks loopback, private, carrier, and multicast IP ranges.
    Returns (is_safe, error_reason).
    """
    if not url:
        return False, "Empty URL provided."

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        return False, f"Malformed URL: {e}"

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False, f"Disallowed URL scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted."

    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80

    if port not in ALLOWED_PORTS:
        return False, f"Disallowed port '{port}'. Only standard web ports (80, 443, 8080) are permitted."

    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname in URL."

    # Prevent localhost keywords
    if hostname.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return False, "Access to localhost or loopback addresses is strictly forbidden."

    # Resolve IP addresses
    try:
        addr_info = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip_str)
            for blocked_net in BLOCKED_IP_NETWORKS:
                if ip_obj in blocked_net:
                    return False, f"Blocked target IP '{ip_str}' is within a private or restricted network."
    except socket.gaierror as e:
        return False, f"Failed to resolve hostname '{hostname}': {e}"
    except Exception as e:
        return False, f"DNS resolution check error: {e}"

    return True, None


def safe_fetch_url(url: str, timeout: int = 10, max_redirects: int = 3) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Fetch web page content with SSRF verification on every redirect step.
    Returns (html_content, status_code, error_message).
    """
    current_url = url
    headers = {"User-Agent": Config.USER_AGENT}

    for _ in range(max_redirects + 1):
        is_safe, reason = is_safe_url(current_url)
        if not is_safe:
            return None, None, f"Security block: {reason}"

        try:
            resp = requests.get(
                current_url,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
                verify=True
            )
            # Handle redirect manually to check target URL
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    break
                current_url = urllib.parse.urljoin(current_url, location)
                continue

            return resp.text, resp.status_code, None
        except requests.exceptions.Timeout:
            return None, None, f"Connection timed out after {timeout} seconds."
        except requests.exceptions.RequestException as e:
            return None, None, f"Network request failed: {str(e)}"

    return None, None, "Exceeded maximum redirect limit."


def is_authorized_recruitment_portal(url: str) -> Tuple[bool, Optional[str]]:
    """Determine if a URL belongs to a verified enterprise recruitment portal (ATS)."""
    if not url:
        return False, None
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        for auth_domain, ats_name in AUTHORIZED_RECRUITMENT_PLATFORMS.items():
            if domain == auth_domain or domain.endswith("." + auth_domain):
                return True, ats_name
    except Exception:
        pass
    return False, None


def check_domain_match(url1: str, url2: str) -> bool:
    """Check if two URLs share the same apex root domain (e.g. careers.google.com and google.com)."""
    if not url1 or not url2:
        return False
    try:
        p1 = urllib.parse.urlparse(url1).netloc.lower().split(":")[-2] if ":" in urllib.parse.urlparse(url1).netloc else urllib.parse.urlparse(url1).netloc.lower()
        p2 = urllib.parse.urlparse(url2).netloc.lower().split(":")[-2] if ":" in urllib.parse.urlparse(url2).netloc else urllib.parse.urlparse(url2).netloc.lower()
        parts1 = p1.split(".")
        parts2 = p2.split(".")
        root1 = ".".join(parts1[-2:]) if len(parts1) >= 2 else p1
        root2 = ".".join(parts2[-2:]) if len(parts2) >= 2 else p2
        return root1 == root2
    except Exception:
        return False


def _normalized_match_text(value: Any) -> str:
    """Normalize titles and company names for strict, punctuation-insensitive comparison."""
    normalized = re.sub(r"[^a-z0-9]+", " ", clean_text(str(value or "")).lower()).strip()
    return re.sub(
        r"\s+(incorporated|inc|limited|ltd|corporation|corp|plc|llc|llp)$",
        "",
        normalized
    ).strip()


def _job_postings_from_page(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Read published JobPosting records from JSON-LD, not arbitrary page text."""
    postings = []

    def collect(value):
        if isinstance(value, list):
            for item in value:
                collect(item)
            return
        if not isinstance(value, dict):
            return
        types = value.get("@type", [])
        if isinstance(types, str):
            types = [types]
        if "JobPosting" in types:
            postings.append(value)
        graph = value.get("@graph")
        if graph:
            collect(graph)

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            collect(json.loads(script.string or script.get_text()))
        except (json.JSONDecodeError, TypeError):
            continue
    return postings


def _published_vacancy(
    soup: BeautifulSoup,
    page_url: str,
    company_name: str,
    job_title: str
) -> Optional[Dict[str, Any]]:
    requested_title = _normalized_match_text(job_title)
    requested_company = _normalized_match_text(company_name)
    if not requested_title or not requested_company:
        return None

    for posting in _job_postings_from_page(soup):
        published_title = clean_text(str(posting.get("title") or "")).strip()
        organization = posting.get("hiringOrganization") or {}
        if isinstance(organization, list):
            organization = organization[0] if organization else {}
        published_company = (
            organization.get("name", "") if isinstance(organization, dict)
            else str(organization)
        )
        if (
            _normalized_match_text(published_title) != requested_title
            or _normalized_match_text(published_company) != requested_company
        ):
            continue

        valid_through = posting.get("validThrough")
        if valid_through:
            try:
                expiry = datetime.fromisoformat(str(valid_through).replace("Z", "+00:00"))
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
                if expiry < datetime.now(timezone.utc):
                    continue
            except ValueError:
                continue

        listing_url = posting.get("url") or page_url
        if not isinstance(listing_url, str) or not listing_url.strip():
            continue
        listing_url = urllib.parse.urljoin(page_url, listing_url.strip())
        vacancy_details = {
            key: posting.get(key)
            for key in ("description", "datePosted", "validThrough", "employmentType", "jobLocation", "identifier")
            if posting.get(key) is not None
        }
        return {
            "evidence_type": "schema.org JobPosting",
            "job_url": listing_url,
            "published_title": published_title,
            "company_name": clean_text(str(published_company)).strip(),
            "vacancy_details": vacancy_details
        }
    return None


def _workday_vacancy(
    portal_url: str,
    company_name: str,
    job_title: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch an exact active posting from a company-linked Workday portal."""
    parsed = urllib.parse.urlparse(portal_url)
    tenant_match = re.fullmatch(
        r"([a-z0-9-]+)\.wd\d+\.myworkdayjobs\.com",
        (parsed.hostname or "").lower()
    )
    if not tenant_match or _normalized_match_text(tenant_match.group(1)) != _normalized_match_text(company_name):
        return None, None

    path_parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    site_id = next(
        (
            part for part in path_parts
            if part.lower() not in {"userhome", "home", "job", "jobs", "apply"}
            and not re.fullmatch(r"[a-z]{2}(?:-[a-z]{2})?", part.lower())
        ),
        None
    )
    if not site_id:
        return None, None

    cxs_base = (
        f"{parsed.scheme}://{parsed.netloc}/wday/cxs/"
        f"{tenant_match.group(1)}/{urllib.parse.quote(site_id, safe='')}"
    )
    search_url = f"{cxs_base}/jobs"
    safe_search, reason = is_safe_url(search_url)
    if not safe_search:
        return None, f"Workday search URL rejected by security policy: {reason}"

    try:
        response = requests.post(
            search_url,
            json={"limit": 20, "offset": 0, "searchText": job_title},
            headers={"User-Agent": Config.USER_AGENT, "Accept": "application/json"},
            timeout=Config.REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
            verify=True
        )
        if response.status_code != 200:
            return None, f"Workday job search returned HTTP {response.status_code}."
        search_data = response.json()
    except (requests.exceptions.RequestException, ValueError) as error:
        return None, f"Workday job search could not be completed: {error}"

    candidates = search_data.get("jobPostings", [])
    if not isinstance(candidates, list):
        return None, "Workday job search returned an invalid response."
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_title = clean_text(str(candidate.get("title") or "")).strip()
        if _normalized_match_text(candidate_title) != _normalized_match_text(job_title):
            continue
        external_path = candidate.get("externalPath")
        if not isinstance(external_path, str) or not external_path.startswith("/job/"):
            continue

        detail_url = cxs_base + external_path
        safe_detail, reason = is_safe_url(detail_url)
        if not safe_detail:
            return None, f"Workday job detail URL rejected by security policy: {reason}"
        try:
            detail_response = requests.get(
                detail_url,
                headers={"User-Agent": Config.USER_AGENT, "Accept": "application/json"},
                timeout=Config.REQUEST_TIMEOUT_SECONDS,
                allow_redirects=False,
                verify=True
            )
            if detail_response.status_code != 200:
                continue
            detail_data = detail_response.json()
        except (requests.exceptions.RequestException, ValueError):
            continue

        if not isinstance(detail_data, dict):
            continue
        posting = detail_data.get("jobPostingInfo") or {}
        if not isinstance(posting, dict):
            continue
        published_title = clean_text(str(posting.get("title") or "")).strip()
        if (
            _normalized_match_text(published_title) != _normalized_match_text(job_title)
            or posting.get("posted") is not True
        ):
            continue
        listing_url = posting.get("externalUrl") or (
            f"{parsed.scheme}://{parsed.netloc}/"
            f"{urllib.parse.quote(site_id, safe='')}{external_path}"
        )
        if not isinstance(listing_url, str):
            continue
        safe_listing, reason = is_safe_url(listing_url)
        listing_host = urllib.parse.urlparse(listing_url).hostname
        if not safe_listing:
            return None, f"Workday job listing URL rejected by security policy: {reason}"
        if listing_host != parsed.hostname:
            return None, "Workday job listing URL does not match the officially linked ATS tenant."

        description_html = posting.get("jobDescription") or ""
        description = clean_text(BeautifulSoup(description_html, "html.parser").get_text(" "))
        vacancy_details = {
            key: posting.get(source_key)
            for key, source_key in (
                ("requisition_id", "jobReqId"),
                ("published_date", "postedOn"),
                ("start_date", "startDate"),
                ("location", "location"),
                ("additional_locations", "additionalLocations"),
                ("employment_type", "timeType")
            )
            if posting.get(source_key) is not None
        }
        if description:
            vacancy_details["description"] = description[:2000]
        return {
            "evidence_type": "Workday job posting",
            "job_url": listing_url,
            "published_title": published_title,
            "company_name": company_name,
            "vacancy_details": vacancy_details
        }, None

    return None, None


def verify_company_and_job(
    company_name: str,
    job_title: str,
    job_description: str,
    application_url: Optional[str] = None,
    official_careers_url: Optional[str] = None,
    official_website_url: Optional[str] = None,
    recruiter_email: Optional[str] = None,
    salary: Optional[str] = None,
    location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform official company website verification according to Section 7 guidelines:
    1. Discover or validate official company website.
    2. Check careers page.
    3. Look for exact or similar vacancy.
    4. Validate application URL.
    5. Formulate evidence, status, and honest explanation.
    """
    evidence = []
    source_urls = []
    official_website = None
    careers_page = None
    matching_vacancy_details = None
    vacancy_matched = False

    # 1. Determine the official website and careers page independently.
    if official_website_url and official_website_url.strip():
        user_url = official_website_url.strip()
        is_safe, reason = is_safe_url(user_url)
        if not is_safe:
            evidence.append(f"User provided official website was rejected by security policy: {reason}")
        else:
            parsed = urllib.parse.urlparse(user_url)
            official_website = f"{parsed.scheme}://{parsed.netloc}"
            source_urls.append(user_url)
            evidence.append(f"Using user-specified official company website: {user_url}")

    if official_careers_url and official_careers_url.strip():
        user_url = official_careers_url.strip()
        is_safe, reason = is_safe_url(user_url)
        if not is_safe:
            evidence.append(f"User provided careers URL was rejected by security policy: {reason}")
        else:
            parsed = urllib.parse.urlparse(user_url)
            official_website = official_website or f"{parsed.scheme}://{parsed.netloc}"
            careers_page = user_url
            source_urls.append(user_url)
            evidence.append(f"Using user-specified company website / careers URL: {user_url}")
    elif not official_website:
        # Search for official company website
        search_res = SearchEngineService.search_company_website(company_name)
        if search_res.get("found"):
            official_website = search_res.get("website_url")
            careers_page = search_res.get("careers_url")
            source_urls.append(official_website)
            if careers_page and careers_page != official_website:
                source_urls.append(careers_page)
            evidence.append(f"Official company website discovered via {search_res.get('source')}: {official_website}")
        else:
            evidence.append(f"Official company website discovery: {search_res.get('status_message')}")

    # 2. Check application URL legitimacy
    is_app_url_suspicious = False
    is_app_url_authorized = False
    authorized_ats_name = None

    if application_url and application_url.strip():
        app_url_clean = application_url.strip()
        is_safe, reason = is_safe_url(app_url_clean)
        if not is_safe:
            is_app_url_suspicious = True
            evidence.append(f"Application URL failed security check: {reason}")
        else:
            source_urls.append(app_url_clean)
            is_ats, ats_name = is_authorized_recruitment_portal(app_url_clean)
            if is_ats:
                is_app_url_authorized = True
                authorized_ats_name = ats_name
                evidence.append(f"Application URL is hosted on an officially recognized recruitment platform: {ats_name} ({app_url_clean})")
            elif official_website and check_domain_match(official_website, app_url_clean):
                evidence.append(f"Application URL domain matches the official company domain ({official_website}).")
            else:
                # Domain mismatch
                if official_website:
                    is_app_url_suspicious = True
                    evidence.append(
                        f"Application URL ({app_url_clean}) does NOT match the official company domain ({official_website}) "
                        "nor any known enterprise applicant tracking system (ATS)."
                    )

    # 3. Inspect the official site and its directly linked job/ATS pages.
    target_fetch_url = careers_page or official_website
    fetch_success = False

    if target_fetch_url:
        pages_to_check = list(dict.fromkeys(
            url for url in (careers_page, official_website) if url
        ))
        if not pages_to_check:
            pages_to_check = [target_fetch_url]
        visited_urls = set()
        linked_ats_urls = set()
        checked_count = 0
        while pages_to_check and checked_count < 16:
            page_url = pages_to_check.pop(0)
            if page_url in visited_urls:
                continue
            visited_urls.add(page_url)
            checked_count += 1

            page_is_official = bool(
                official_website and check_domain_match(official_website, page_url)
            )
            if not page_is_official and not is_authorized_recruitment_portal(page_url)[0]:
                continue
            html_content, status_code, err = safe_fetch_url(
                page_url, timeout=Config.REQUEST_TIMEOUT_SECONDS
            )
            if err:
                evidence.append(f"Careers page check note: Unable to fetch {page_url} ({err}).")
                continue
            if status_code != 200 or not html_content:
                continue

            fetch_success = True
            source_urls.append(page_url)
            soup = BeautifulSoup(html_content, "html.parser")
            matched_posting = _published_vacancy(soup, page_url, company_name, job_title)
            if not matched_posting and page_url in linked_ats_urls:
                matched_posting, ats_error = _workday_vacancy(
                    page_url, company_name, job_title
                )
                if ats_error:
                    evidence.append(ats_error)
            if matched_posting:
                listing_url = matched_posting["job_url"]
                listing_safe, listing_error = is_safe_url(listing_url)
                listing_is_official = (
                    official_website and check_domain_match(official_website, listing_url)
                )
                listing_is_linked_ats = page_url in linked_ats_urls
                if listing_safe and (listing_is_official or listing_is_linked_ats):
                    vacancy_matched = True
                    matching_vacancy_details = matched_posting
                    matching_vacancy_details["job_url"] = listing_url
                    source_urls.append(listing_url)
                    evidence.append(
                        f"Published JobPosting for '{matched_posting['published_title']}' at "
                        f"{matched_posting['company_name']} was confirmed: {listing_url}"
                    )
                    break

            # ATS pages are only considered when linked from a company-domain page.
            if page_is_official:
                for anchor in soup.find_all("a", href=True):
                    linked_url = urllib.parse.urljoin(page_url, anchor["href"].strip())
                    parsed_link = urllib.parse.urlparse(linked_url)
                    link_hint = f"{anchor.get_text(' ', strip=True)} {parsed_link.path}".lower()
                    linked_ats = is_authorized_recruitment_portal(linked_url)[0]
                    job_link = re.search(
                        r"\b(job|jobs|career|careers|vacan|requisition|position)\b",
                        link_hint
                    )
                    if linked_ats or job_link:
                        if linked_url not in visited_urls and linked_url not in pages_to_check:
                            if linked_ats:
                                linked_ats_urls.add(linked_url)
                                pages_to_check.insert(0, linked_url)
                            else:
                                pages_to_check.append(linked_url)

    # 4. Synthesize Verification Status according to Section 7 Rules
    # Statuses: VERIFIED GENUINE, LIKELY SUSPICIOUS, UNVERIFIED
    # Important rules:
    # - A job not appearing on official website does NOT automatically mean fake.
    # - Search results alone are not proof.
    # - Website failure must be displayed as website-check failure, not proof no vacancy exists.
    # - Distinguish ML predictions from official verification.

    # Check for strong impersonation/fraud indicators
    strong_fraud_flags = []
    if recruiter_email and "@" in recruiter_email:
        email_domain = recruiter_email.split("@")[-1].lower()
        if email_domain in ("gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "mail.com") and official_website:
            strong_fraud_flags.append(
                f"Recruiter uses free email (@{email_domain}) while claiming to represent {company_name}."
            )
    if is_app_url_suspicious and not is_app_url_authorized:
        strong_fraud_flags.append("Application URL points to an unverified or mismatched external domain.")

    if vacancy_matched:
        verification_status = "VERIFIED GENUINE"
        explanation = (
            "A specific, active vacancy with an exact title and company match was confirmed in a published "
            "JobPosting record on the official company site or its linked recruitment portal."
        )
    elif len(strong_fraud_flags) >= 1:
        verification_status = "LIKELY SUSPICIOUS"
        explanation = (
            f"Strong discrepancy or impersonation flags were detected: " + "; ".join(strong_fraud_flags) +
            ". Legitimate organizations typically direct applications through their official domain or verified ATS."
        )
    else:
        verification_status = "UNVERIFIED"
        if not target_fetch_url:
            explanation = (
                "Insufficient information was available to locate the company's official careers website. "
                "The recruitment could not be verified online. Providing an official careers URL will allow automatic verification."
            )
        elif not fetch_success:
            explanation = (
                f"The official company web portal could not be reached or timed out during verification. "
                "This indicates a website connection limitation, not proof that the job is fraudulent."
            )
        else:
            explanation = (
                f"The company website ({target_fetch_url}) was reached, but the advertised role '{job_title}' was not "
                "found on the scanned page. This posting is unverified. Note that unverified listings may be hosted on "
                "separate job boards or dynamic portals, so this alone does not prove the posting is fake."
            )

    if vacancy_matched:
        official_verification_status = "VERIFIED GENUINE"
    elif strong_fraud_flags:
        official_verification_status = "LIKELY SUSPICIOUS"
    elif target_fetch_url and not fetch_success:
        official_verification_status = "UNVERIFIED - WEBSITE CHECK FAILED"
    else:
        official_verification_status = "UNVERIFIED"

    return {
        "verification_status": verification_status,
        "official_verification_status": official_verification_status,
        "official_website_url": official_website,
        "careers_page_url": careers_page or target_fetch_url,
        "vacancy_matched": bool(vacancy_matched),
        "matching_vacancy_details": matching_vacancy_details,
        "evidence": evidence,
        "source_urls": list(dict.fromkeys(source_urls)),
        "explanation": explanation
    }
