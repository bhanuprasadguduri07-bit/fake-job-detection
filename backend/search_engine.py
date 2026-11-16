"""
Search Engine API integration for discovering official company websites.
Supports Google Programmable Search, Bing Web Search API, and a clean fallback
discovery method. Never fabricates search results.
"""

import os
import re
import urllib.parse
from typing import Dict, Any, Optional, List
import requests
from config import Config


# Well-known verified corporate domains for instant trusted resolution
TRUSTED_CORPORATE_DOMAINS = {
    "google": ("Google LLC", "https://www.google.com", "https://www.google.com/about/careers"),
    "microsoft": ("Microsoft Corporation", "https://www.microsoft.com", "https://careers.microsoft.com"),
    "amazon": ("Amazon.com, Inc.", "https://www.amazon.com", "https://www.amazon.jobs"),
    "apple": ("Apple Inc.", "https://www.apple.com", "https://www.apple.com/careers"),
    "meta": ("Meta Platforms, Inc.", "https://about.meta.com", "https://www.metacareers.com"),
    "tcs": ("Tata Consultancy Services", "https://www.tcs.com", "https://www.tcs.com/careers"),
    "infosys": ("Infosys Limited", "https://www.infosys.com", "https://www.infosys.com/careers"),
    "wipro": ("Wipro Limited", "https://www.wipro.com", "https://careers.wipro.com"),
    "ibm": ("IBM", "https://www.ibm.com", "https://www.ibm.com/careers"),
    "netflix": ("Netflix, Inc.", "https://www.netflix.com", "https://jobs.netflix.com"),
    "accenture": ("Accenture", "https://www.accenture.com", "https://www.accenture.com/careers"),
    "salesforce": ("Salesforce, Inc.", "https://www.salesforce.com", "https://careers.salesforce.com"),
    "adobe": ("Adobe Inc.", "https://www.adobe.com", "https://careers.adobe.com"),
    "intel": ("Intel Corporation", "https://www.intel.com", "https://jobs.intel.com"),
    "cisco": ("Cisco Systems", "https://www.cisco.com", "https://jobs.cisco.com")
}


class SearchEngineService:
    """Discovers official company websites using configured search APIs or public discovery."""

    @staticmethod
    def is_api_configured() -> bool:
        """Check if any real external search API is configured in environment."""
        has_google = bool(Config.GOOGLE_SEARCH_API_KEY and Config.GOOGLE_SEARCH_ENGINE_ID)
        has_bing = bool(Config.BING_SEARCH_API_KEY)
        return has_google or has_bing

    @classmethod
    def search_company_website(cls, company_name: str) -> Dict[str, Any]:
        """
        Search for the official company website.
        Returns:
        {
            "found": bool,
            "company_name": str,
            "website_url": Optional[str],
            "careers_url": Optional[str],
            "source": str,
            "status_message": str
        }
        """
        cleaned_company = (company_name or "").strip()
        if not cleaned_company:
            return {
                "found": False,
                "company_name": "",
                "website_url": None,
                "careers_url": None,
                "source": "None",
                "status_message": "No company name provided for website search."
            }

        # Check known trusted corporate mapping first for accuracy
        normalized_name = re.sub(r"[^a-zA-Z0-9]", "", cleaned_company.lower())
        if normalized_name in TRUSTED_CORPORATE_DOMAINS:
            official_name, web_url, car_url = TRUSTED_CORPORATE_DOMAINS[normalized_name]
            return {
                "found": True,
                "company_name": official_name,
                "website_url": web_url,
                "careers_url": car_url,
                "source": "Verified Corporate Registry",
                "status_message": f"Resolved official corporate domain for '{official_name}'."
            }

        # 1. Google Custom Search JSON API
        if Config.GOOGLE_SEARCH_API_KEY and Config.GOOGLE_SEARCH_ENGINE_ID:
            res = cls._search_google(cleaned_company)
            if res.get("found"):
                return res

        # 2. Bing Web Search API
        if Config.BING_SEARCH_API_KEY:
            res = cls._search_bing(cleaned_company)
            if res.get("found"):
                return res

        # 3. Public Web Discovery via DuckDuckGo HTML (no API key required)
        res_ddg = cls._search_duckduckgo(cleaned_company)
        if res_ddg.get("found"):
            return res_ddg

        # If all search methods are unavailable or yield no result
        if not cls.is_api_configured():
            msg = (
                f"Automated search API not configured (GOOGLE_SEARCH_API_KEY / BING_SEARCH_API_KEY not set in .env). "
                f"Could not automatically locate official website for '{cleaned_company}'. "
                f"Please provide the company's official website URL directly in the form."
            )
        else:
            msg = f"Search engine returned no verified official domain for company '{cleaned_company}'."

        return {
            "found": False,
            "company_name": cleaned_company,
            "website_url": None,
            "careers_url": None,
            "source": "Search Engine API",
            "status_message": msg
        }

    @staticmethod
    def _search_google(company_name: str) -> Dict[str, Any]:
        """Query Google Custom Search JSON API."""
        try:
            query = f"{company_name} official website careers"
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": Config.GOOGLE_SEARCH_API_KEY,
                "cx": Config.GOOGLE_SEARCH_ENGINE_ID,
                "q": query,
                "num": 3
            }
            resp = requests.get(url, params=params, timeout=Config.REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])
                if items:
                    first_link = items[0].get("link")
                    parsed = urllib.parse.urlparse(first_link)
                    base_url = f"{parsed.scheme}://{parsed.netloc}"
                    return {
                        "found": True,
                        "company_name": company_name,
                        "website_url": base_url,
                        "careers_url": first_link,
                        "source": "Google Custom Search API",
                        "status_message": f"Discovered official website via Google Custom Search: {base_url}"
                    }
        except Exception as e:
            print(f"Google Search API error: {e}")
        return {"found": False}

    @staticmethod
    def _search_bing(company_name: str) -> Dict[str, Any]:
        """Query Bing Web Search API."""
        try:
            endpoint = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": Config.BING_SEARCH_API_KEY}
            params = {"q": f"{company_name} official careers website", "count": 3}
            resp = requests.get(endpoint, headers=headers, params=params, timeout=Config.REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                data = resp.json()
                web_pages = data.get("webPages", {}).get("value", [])
                if web_pages:
                    first_link = web_pages[0].get("url")
                    parsed = urllib.parse.urlparse(first_link)
                    base_url = f"{parsed.scheme}://{parsed.netloc}"
                    return {
                        "found": True,
                        "company_name": company_name,
                        "website_url": base_url,
                        "careers_url": first_link,
                        "source": "Bing Web Search API",
                        "status_message": f"Discovered official website via Bing Web Search: {base_url}"
                    }
        except Exception as e:
            print(f"Bing Search API error: {e}")
        return {"found": False}

    @staticmethod
    def _search_duckduckgo(company_name: str) -> Dict[str, Any]:
        """Perform fallback discovery using DuckDuckGo HTML endpoint."""
        try:
            headers = {"User-Agent": Config.USER_AGENT}
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(company_name + ' official website')}"
            resp = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", class_="result__url"):
                    href = a.get("href", "").strip()
                    # DuckDuckGo redirect link parsing
                    if "uddg=" in href:
                        real_url = urllib.parse.unquote(href.split("uddg=")[-1].split("&")[0])
                        parsed = urllib.parse.urlparse(real_url)
                        if parsed.netloc and not any(skip in parsed.netloc for skip in ["duckduckgo", "wikipedia"]):
                            base_url = f"{parsed.scheme}://{parsed.netloc}"
                            return {
                                "found": True,
                                "company_name": company_name,
                                "website_url": base_url,
                                "careers_url": f"{base_url}/careers",
                                "source": "DuckDuckGo Web Discovery",
                                "status_message": f"Located potential company website: {base_url}"
                            }
        except Exception as e:
            print(f"DuckDuckGo fallback search error: {e}")
        return {"found": False}

