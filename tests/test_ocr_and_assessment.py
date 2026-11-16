import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from backend.assessment import combine_recruitment_assessment
from backend.image_processor import preprocess_image_for_ocr
from backend.ocr_engine import extract_entities_from_text, perform_ocr
from backend.website_verifier import verify_company_and_job


class OCRFieldMappingTests(unittest.TestCase):
    def test_poster_with_all_fields(self):
        fields = extract_entities_from_text(
            "Company Name: Accenture\n"
            "Job Title: Associate Engineer - Backend\n"
            "Work Experience: 1-2 Years\n"
            "Required Skills: Java 11+, Spring Boot\n"
            "Education / Qualification: B.E / B.Tech / MCA\n"
            "Job Location: Bengaluru, India\n"
            "Salary: As per company standards\n"
            "Job Description: Develop and maintain backend applications.\n"
            "Recruiter Email: jobs@example.com\n"
            "Contact Number: +91 98765 43210\n"
            "Application URL: https://example.com/jobs/123\n"
            "Official Website: https://example.com"
        )
        self.assertEqual(fields["company_name"], "Accenture")
        self.assertEqual(fields["job_title"], "Associate Engineer - Backend")
        self.assertEqual(fields["experience_requirements"], "1-2 Years")
        self.assertEqual(fields["required_skills"], "Java 11+, Spring Boot")
        self.assertEqual(fields["education"], "B.E / B.Tech / MCA")
        self.assertEqual(fields["job_location"], "Bengaluru, India")
        self.assertEqual(fields["salary"], "As per company standards")
        self.assertEqual(fields["job_description"], "Develop and maintain backend applications.")
        self.assertEqual(fields["recruiter_email"], "jobs@example.com")
        self.assertEqual(fields["contact_number"], "+91 98765 43210")
        self.assertEqual(fields["application_url"], "https://example.com/jobs/123")
        self.assertEqual(fields["official_website"], "https://example.com")

    def test_experience_and_skills_stay_in_separate_sections(self):
        fields = extract_entities_from_text(
            "Experience: 3+ years\nRequired Skills:\n- Python\n- SQL\nEducation: MSc"
        )
        self.assertEqual(fields["experience_requirements"], "3+ years")
        self.assertEqual(fields["required_skills"], "Python, SQL")
        self.assertEqual(fields["education"], "MSc")

    def test_location_beside_map_pin_icon(self):
        fields = extract_entities_from_text("Experience: 2 years\n📍 Bengaluru, India\nSkills: Java")
        self.assertEqual(fields["job_location"], "Bengaluru, India")
        self.assertEqual(fields["experience_requirements"], "2 years")
        self.assertEqual(fields["required_skills"], "Java")

    def test_map_pin_on_own_line_does_not_append_city_to_education(self):
        fields = extract_entities_from_text("Education: BSc\n📍\nHyderabad, India\nSkills: Java")
        self.assertEqual(fields["education"], "BSc")
        self.assertEqual(fields["job_location"], "Hyderabad, India")

    def test_pan_india_location(self):
        self.assertEqual(
            extract_entities_from_text("Location: PAN India")["job_location"],
            "PAN India"
        )

    def test_missing_location_is_left_empty(self):
        self.assertEqual(
            extract_entities_from_text("Job Title: Engineer\nSkills: Java")["job_location"],
            ""
        )

    def test_small_poor_quality_image_is_upscaled_and_preprocessed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "small.png")
            Image.new("RGB", (80, 40), color=(150, 150, 150)).save(path)
            processed = preprocess_image_for_ocr(path)
        self.assertEqual(processed.mode, "L")
        self.assertGreater(max(processed.size), 80)

    def test_missing_education_is_left_empty(self):
        fields = extract_entities_from_text("Job Title: Analyst\nExperience: 1 year\nSkills: Excel")
        self.assertEqual(fields["education"], "")
        self.assertNotIn("Education", fields["required_skills"])

    def test_unreadable_education_text_is_left_empty(self):
        fields = extract_entities_from_text(
            "Education:\n"
            "devexceent � Coffatorate with teams on clent peosects "
            "ai OL / 0. tech!) MA ContnQuite to celiveritus"
        )
        self.assertEqual(fields["education"], "")

    def test_multiple_skills_are_aggregated_without_other_fields(self):
        fields = extract_entities_from_text(
            "Skills:\n- Java 11+\n- Spring Boot\n- REST APIs\n- SQL / NoSQL\n"
            "Experience: 1-2 Years\nEducation: MCA"
        )
        self.assertEqual(fields["required_skills"], "Java 11+, Spring Boot, REST APIs, SQL / NoSQL")
        self.assertEqual(fields["experience_requirements"], "1-2 Years")
        self.assertEqual(fields["education"], "MCA")

    def test_salary_and_experience_on_same_line(self):
        fields = extract_entities_from_text("Salary: INR 8-10 LPA | Experience: 2-4 years")
        self.assertEqual(fields["salary"], "INR 8-10 LPA")
        self.assertEqual(fields["experience_requirements"], "2-4 years")

    def test_long_description_is_kept_intact(self):
        description = (
            "Develop reliable services and APIs. Collaborate with engineering teams. "
            "Review code, troubleshoot production issues, and improve database performance."
        )
        fields = extract_entities_from_text(
            f"Job Description:\n{description}\nExperience: 3 years\nSkills: Python"
        )
        self.assertEqual(fields["job_description"], description)
        self.assertEqual(fields["experience_requirements"], "3 years")
        self.assertEqual(fields["required_skills"], "Python")

    def test_unlabeled_job_duties_are_extracted_without_promotional_copy(self):
        fields = extract_entities_from_text(
            "Why Infosys?\n"
            "Global leader in IT services\n"
            "Excellent learning opportunities\n"
            "Work on digital solutions for clients\n"
            "Participate in software design and development\n"
            "Perform software testing and validation\n"
            "Collaborate with teams on client projects\n"
            "Contribute to delivering high-quality solutions\n"
            "Education: B.E / B.Tech"
        )
        self.assertEqual(
            fields["job_description"],
            "Work on digital solutions for clients "
            "Participate in software design and development "
            "Perform software testing and validation "
            "Collaborate with teams on client projects "
            "Contribute to delivering high-quality solutions"
        )

    def test_location_is_not_guessed_from_address_or_url(self):
        fields = extract_entities_from_text(
            "Company address: 12 Bengaluru Road, India\nhttps://jobs.example.com/location"
        )
        self.assertEqual(fields["job_location"], "")
        self.assertEqual(
            extract_entities_from_text("Company address\nBengaluru, India")["job_location"],
            ""
        )

    def test_unavailable_tesseract_returns_editable_empty_fields(self):
        with patch("backend.ocr_engine.find_tesseract_binary", return_value=None):
            result = perform_ocr("unused.png")
        self.assertFalse(result["success"])
        self.assertIn("company_name", result["extracted_fields"])
        self.assertEqual(result["extracted_fields"]["education"], "")


class FinalAssessmentTests(unittest.TestCase):
    def test_likely_genuine_and_verified(self):
        status, _ = combine_recruitment_assessment(
            {"prediction_label": "Likely Genuine", "is_fake": False},
            "Verified on Official Website"
        )
        self.assertEqual(status, "VERIFIED GENUINE")

    def test_likely_fake_and_not_found(self):
        status, reason = combine_recruitment_assessment(
            {"prediction_label": "Likely Fake", "is_fake": True},
            "Not Found on Official Website"
        )
        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("could not be confidently verified", reason)

    def test_likely_genuine_and_not_found_stays_unverified(self):
        status, _ = combine_recruitment_assessment(
            {"prediction_label": "Likely Genuine", "is_fake": False},
            "Not Found on Official Website"
        )
        self.assertEqual(status, "UNVERIFIED")

    def test_likely_fake_but_verified_requires_review(self):
        status, _ = combine_recruitment_assessment(
            {"prediction_label": "Likely Fake", "is_fake": True},
            "Verified on Official Website"
        )
        self.assertEqual(status, "MANUAL REVIEW REQUIRED")

    def test_website_failure_is_unverified(self):
        status, reason = combine_recruitment_assessment(
            {"prediction_label": "Likely Genuine", "is_fake": False},
            "Website Check Failed"
        )
        self.assertEqual(status, "UNVERIFIED - WEBSITE CHECK FAILED")
        self.assertIn("not evidence", reason)

    def test_strong_fraud_evidence_conflicting_with_ml_requires_review(self):
        status, _ = combine_recruitment_assessment(
            {"prediction_label": "Likely Genuine", "is_fake": False},
            "LIKELY SUSPICIOUS"
        )
        self.assertEqual(status, "MANUAL REVIEW REQUIRED")


class OfficialVerificationTests(unittest.TestCase):
    def verify_with_page(self, html, status=200, error=None):
        with (
            patch("backend.website_verifier.is_safe_url", return_value=(True, None)),
            patch("backend.website_verifier.safe_fetch_url", return_value=(html, status, error))
        ):
            return verify_company_and_job(
                company_name="Acme",
                job_title="Backend Engineer",
                job_description="Build backend services.",
                official_website_url="https://example.com"
            )

    def test_matching_vacancy_is_verified(self):
        page = """
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "JobPosting",
          "title": "Backend Engineer",
          "url": "https://example.com/careers/backend-engineer-123",
          "hiringOrganization": {"@type": "Organization", "name": "Acme"},
          "description": "Build backend services."
        }
        </script>
        """
        result = self.verify_with_page(page)
        self.assertEqual(result["official_verification_status"], "VERIFIED GENUINE")
        self.assertEqual(result["matching_vacancy_details"]["published_title"], "Backend Engineer")
        self.assertEqual(
            result["matching_vacancy_details"]["job_url"],
            "https://example.com/careers/backend-engineer-123"
        )

    def test_homepage_title_or_partial_keywords_do_not_verify_vacancy(self):
        page = "<html><body>Welcome to Acme. Backend openings for engineers.</body></html>"
        result = self.verify_with_page(page)
        self.assertEqual(result["official_verification_status"], "UNVERIFIED")
        self.assertFalse(result["vacancy_matched"])

    def test_accenture_back_bancher_remains_unverified_without_listing(self):
        page = "<html><body>Accenture careers: Back Office Banker opportunities.</body></html>"
        with (
            patch("backend.website_verifier.is_safe_url", return_value=(True, None)),
            patch("backend.website_verifier.safe_fetch_url", return_value=(page, 200, None))
        ):
            result = verify_company_and_job(
                company_name="Accenture",
                job_title="Back Bancher",
                job_description="",
                official_website_url="https://www.accenture.com"
            )
        self.assertEqual(result["official_verification_status"], "UNVERIFIED")
        self.assertFalse(result["vacancy_matched"])

    def test_accenture_unknown_position_remains_unverified(self):
        with (
            patch("backend.website_verifier.is_safe_url", return_value=(True, None)),
            patch(
                "backend.website_verifier.safe_fetch_url",
                return_value=("<html><body>Accenture careers</body></html>", 200, None)
            )
        ):
            result = verify_company_and_job(
                company_name="Accenture",
                job_title="Unknown Position XYZ123",
                job_description="",
                official_website_url="https://www.accenture.com"
            )
        self.assertEqual(result["official_verification_status"], "UNVERIFIED")
        self.assertFalse(result["vacancy_matched"])

    def test_website_fetch_error_is_distinct_from_not_found(self):
        result = self.verify_with_page(None, status=None, error="connection timed out")
        self.assertEqual(
            result["official_verification_status"],
            "UNVERIFIED - WEBSITE CHECK FAILED"
        )

    def test_no_discovered_website_is_unverified(self):
        with (
            patch(
                "backend.website_verifier.SearchEngineService.search_company_website",
                return_value={"found": False, "status_message": "No website found"}
            ),
            patch("backend.website_verifier.is_safe_url", return_value=(True, None))
        ):
            result = verify_company_and_job(
                company_name="Acme",
                job_title="Backend Engineer",
                job_description="Build backend services."
            )
        self.assertEqual(result["official_verification_status"], "UNVERIFIED")

    def test_authorized_ats_requires_official_link_and_published_listing(self):
        listing = """
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "JobPosting",
          "title": "Backend Engineer",
          "url": "https://boards.greenhouse.io/acme/jobs/123",
          "hiringOrganization": {"@type": "Organization", "name": "Acme"}
        }
        </script>
        """
        official = '<a href="https://boards.greenhouse.io/acme/jobs/123">Backend Engineer</a>'

        def fetch(url, **_kwargs):
            if "greenhouse.io" in url:
                return listing, 200, None
            return official, 200, None

        with (
            patch("backend.website_verifier.is_safe_url", return_value=(True, None)),
            patch("backend.website_verifier.safe_fetch_url", side_effect=fetch)
        ):
            result = verify_company_and_job(
                company_name="Acme",
                job_title="Backend Engineer",
                job_description="",
                official_website_url="https://example.com"
            )
        self.assertEqual(result["official_verification_status"], "VERIFIED GENUINE")
        self.assertEqual(
            result["matching_vacancy_details"]["job_url"],
            "https://boards.greenhouse.io/acme/jobs/123"
        )

    def test_workday_posting_is_verified_only_when_linked_from_company_site(self):
        portal_url = "https://acme.wd103.myworkdayjobs.com/AcmeCareers/userHome"
        posting = {
            "evidence_type": "Workday job posting",
            "job_url": "https://acme.wd103.myworkdayjobs.com/AcmeCareers/job/remote/backend-engineer_R123",
            "published_title": "Backend Engineer",
            "company_name": "Acme",
            "vacancy_details": {"requisition_id": "R123"}
        }

        def fetch(url, **_kwargs):
            if url == "https://example.com":
                return (
                    f'<a href="{portal_url}">Search our careers</a>',
                    200,
                    None
                )
            return "<html><body>Accenture</body></html>", 200, None

        with (
            patch("backend.website_verifier.is_safe_url", return_value=(True, None)),
            patch("backend.website_verifier.safe_fetch_url", side_effect=fetch),
            patch(
                "backend.website_verifier._workday_vacancy",
                return_value=(posting, None)
            ) as workday_check
        ):
            result = verify_company_and_job(
                company_name="Acme",
                job_title="Backend Engineer",
                job_description="",
                official_website_url="https://example.com"
            )
        self.assertEqual(result["official_verification_status"], "VERIFIED GENUINE")
        self.assertEqual(result["matching_vacancy_details"]["job_url"], posting["job_url"])
        workday_check.assert_called_once_with(portal_url, "Acme", "Backend Engineer")

    def test_workday_application_url_alone_does_not_verify_vacancy(self):
        with (
            patch("backend.website_verifier.is_safe_url", return_value=(True, None)),
            patch(
                "backend.website_verifier.safe_fetch_url",
                return_value=("<html><body>Welcome to Acme</body></html>", 200, None)
            ),
            patch("backend.website_verifier._workday_vacancy") as workday_check
        ):
            result = verify_company_and_job(
                company_name="Acme",
                job_title="Backend Engineer",
                job_description="",
                official_website_url="https://example.com",
                application_url="https://acme.wd103.myworkdayjobs.com/AcmeCareers/job/123"
            )
        self.assertEqual(result["official_verification_status"], "UNVERIFIED")
        workday_check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
