import importlib
import io
import os
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from PIL import Image

from backend.ocr_engine import extract_entities_from_text
from backend.scheduler import MonitoringScheduler
from config import Config


class FlaskWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_directory = tempfile.TemporaryDirectory()
        Config.DATABASE_PATH = os.path.join(cls.temp_directory.name, "test.db")
        Config.UPLOAD_FOLDER = os.path.join(cls.temp_directory.name, "uploads")
        with patch.object(MonitoringScheduler, "start_scheduler"):
            cls.app_module = importlib.import_module("app")
            cls.application = cls.app_module.create_app()
        cls.application.config.update(
            TESTING=True,
            START_MONITORING_SCHEDULER=False
        )
        cls.client = cls.application.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.temp_directory.cleanup()

    def verification_result(self):
        return {
            "verification_status": "UNVERIFIED",
            "official_verification_status": "Not Found on Official Website",
            "official_website_url": "https://example.com",
            "careers_page_url": "https://example.com/careers",
            "vacancy_matched": False,
            "matching_vacancy_details": None,
            "evidence": ["Test verification evidence"],
            "source_urls": ["https://example.com/careers"],
            "explanation": "No matching job was found on the checked page."
        }

    def test_image_api_and_reviewed_poster_submission(self):
        raw_text = (
            "Company Name: Acme Systems\n"
            "Job Title: Backend Engineer\n"
            "Experience: 3 years\n"
            "Skills:\n- Python\n- SQL\n"
            "Education: BSc Computer Science\n"
            "📍 Remote\n"
            "Salary: $90,000 per year\n"
            "Job Description: Build APIs and maintain backend services."
        )
        extracted = extract_entities_from_text(raw_text)

        def save_upload(file_storage):
            image_path = os.path.join(Config.UPLOAD_FOLDER, "poster.png")
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            file_storage.save(image_path)
            return "/static/uploads/poster.png", image_path, os.path.getsize(image_path)

        image_buffer = io.BytesIO()
        Image.new("RGB", (200, 100), "white").save(image_buffer, format="PNG")
        image_buffer.seek(0)
        with (
            patch.object(self.app_module, "save_uploaded_image", side_effect=save_upload),
            patch.object(
                self.app_module,
                "perform_ocr",
                return_value={
                    "success": True,
                    "raw_text": raw_text,
                    "engine_used": "Tesseract OCR",
                    "error_message": None,
                    "extracted_fields": extracted
                }
            ),
            patch.object(
                self.app_module,
                "verify_company_and_job",
                return_value=self.verification_result()
            )
        ):
            response = self.client.post(
                "/api/ocr/extract",
                data={"poster_image": (image_buffer, "poster.png")},
                content_type="multipart/form-data"
            )
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertTrue(payload["success"])
            self.assertEqual(payload["extracted_fields"]["experience_requirements"], "3 years")
            self.assertEqual(payload["extracted_fields"]["required_skills"], "Python, SQL")
            self.assertEqual(payload["extracted_fields"]["education"], "BSc Computer Science")
            self.assertEqual(payload["extracted_fields"]["job_location"], "Remote")

            form = {
                **{key: extracted.get(key, "") for key in (
                    "company_name", "job_title", "job_description", "salary",
                    "job_location", "required_skills", "experience_requirements",
                    "education", "recruiter_email", "contact_number", "application_url",
                    "official_website_url", "official_careers_url"
                )},
                "image_path": payload["file_path"],
                "image_name": payload["file_name"],
                "raw_text": payload["raw_text"],
                "ocr_engine": payload["ocr_engine"]
            }
            result = self.client.post("/detect/image/submit", data=form, follow_redirects=True)
            self.assertEqual(result.status_code, 200)
            self.assertIn(b"OCR Extracted Details", result.data)
            self.assertIn(b"Raw OCR text", result.data)

        from database import get_db_connection
        connection = get_db_connection()
        submission = connection.execute(
            "SELECT experience_requirements, required_skills, education, job_location "
            "FROM job_submissions WHERE submission_type = 'poster' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        stored_ocr = connection.execute(
            "SELECT extracted_raw_text FROM uploaded_posters ORDER BY id DESC LIMIT 1"
        ).fetchone()
        connection.close()
        self.assertEqual(submission["experience_requirements"], "3 years")
        self.assertEqual(submission["required_skills"], "Python, SQL")
        self.assertEqual(submission["education"], "BSc Computer Science")
        self.assertEqual(submission["job_location"], "Remote")
        self.assertEqual(stored_ocr["extracted_raw_text"], raw_text)

    def test_text_submission_uses_the_same_fields_and_results_page(self):
        with patch.object(
            self.app_module,
            "verify_company_and_job",
            return_value=self.verification_result()
        ):
            response = self.client.post(
                "/detect/text",
                data={
                    "company_name": "Acme Systems",
                    "job_title": "Backend Engineer",
                    "job_description": "Build APIs and maintain backend services.",
                    "experience_requirements": "3 years",
                    "required_skills": "Python, SQL",
                    "education": "BSc Computer Science",
                    "job_location": "Remote",
                    "salary": "$90,000 per year",
                    "recruiter_email": "jobs@example.com",
                    "contact_number": "+1 212 555 0199",
                    "application_url": "https://example.com/jobs/123",
                    "official_website_url": "https://example.com"
                },
                follow_redirects=True
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Machine Learning Prediction", response.data)
        self.assertIn(b"Official Company Website Verification", response.data)
        self.assertIn(b"Final Recruitment Status", response.data)

    def test_empty_standalone_verification_does_not_render_a_previous_result(self):
        response = self.client.get("/verify")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"Verification Result", response.data)

        response = self.client.post(
            "/verify",
            data={"company_name": "", "job_title": ""}
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"Verification Result", response.data)
        self.assertNotIn(b"VERIFIED GENUINE", response.data)
        self.assertIn(b'name="company_name" class="form-control" required', response.data)
        self.assertIn(b'name="job_title" class="form-control" required', response.data)

    def test_legacy_keyword_match_is_not_displayed_as_verified(self):
        legacy_result = {
            "verification_status": "VERIFIED GENUINE",
            "official_verification_status": "Verified on Official Website",
            "official_website_url": "https://example.com",
            "careers_page_url": "https://example.com/careers",
            "vacancy_matched": True,
            "matching_vacancy_details": {
                "matched_title": "Backend Engineer",
                "match_type": "Fuzzy Keyword Match",
                "source_url": "https://example.com"
            },
            "evidence": ["Similar vacancy matching title keywords located on official portal."],
            "source_urls": ["https://example.com"],
            "explanation": "Old homepage keyword match."
        }
        with patch.object(
            self.app_module,
            "verify_company_and_job",
            return_value=legacy_result
        ):
            response = self.client.post(
                "/detect/text",
                data={
                    "company_name": "Acme Systems",
                    "job_title": "Backend Engineer",
                    "job_description": "Build APIs and maintain backend services."
                },
                follow_redirects=True
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"UNVERIFIED", response.data)
        self.assertNotIn(b"Verified on Official Website", response.data)
        self.assertNotIn(b"Fuzzy Keyword Match", response.data)

    def test_result_is_only_available_to_its_submitting_user(self):
        from database import get_db_connection

        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("result-owner", "result-owner@example.com", "unused")
        )
        owner_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO job_submissions (user_id, company_name, job_title, job_description) "
            "VALUES (?, ?, ?, ?)",
            (owner_id, "Private Company", "Private Role", "Private description")
        )
        submission_id = cursor.lastrowid
        connection.commit()
        connection.close()

        with self.client.session_transaction() as browser_session:
            browser_session["user_id"] = owner_id + 1000
        response = self.client.get(f"/result/{submission_id}")
        self.assertEqual(response.status_code, 404)
        with self.client.session_transaction() as browser_session:
            browser_session.clear()

    def test_concurrent_detection_dashboard_and_monitoring_writes(self):
        from database import get_db_connection

        marker = uuid.uuid4().hex[:10]
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (f"lock-test-{marker}", f"lock-test-{marker}@example.com", "unused")
        )
        user_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO monitored_websites (user_id, company_name, careers_url) "
            "VALUES (?, ?, ?)",
            (user_id, "Concurrency Test", "https://example.com/careers")
        )
        site_id = cursor.lastrowid
        connection.commit()
        connection.close()

        barrier = threading.Barrier(8)
        prediction = {
            "prediction_label": "Likely Genuine",
            "is_fake": False,
            "confidence_score": 0.9,
            "fake_probability": 0.1,
            "suspicious_indicators": [],
            "explanation": "Concurrent workflow test",
            "metrics": {}
        }

        def submit_detection(index):
            client = self.application.test_client()
            barrier.wait(timeout=10)
            response = client.post(
                "/detect/text",
                data={
                    "company_name": f"Concurrent Test {marker} {index}",
                    "job_title": "Backend Engineer",
                    "job_description": "Build and maintain backend services."
                },
                follow_redirects=True
            )
            return response.status_code

        def read_dashboard():
            client = self.application.test_client()
            with client.session_transaction() as browser_session:
                browser_session["user_id"] = user_id
            barrier.wait(timeout=10)
            return client.get("/dashboard").status_code

        def monitor_site():
            barrier.wait(timeout=10)
            return MonitoringScheduler.check_single_website(
                site_id, user_id, "Concurrency Test", "https://example.com/careers"
            )

        def crawl_site(_url):
            threading.Event().wait(0.1)
            return ([{
                "job_title": f"Monitored Role {marker}",
                "job_location": "Remote",
                "department": "Engineering",
                "salary": "",
                "description": "Build backend services.",
                "job_url": "https://example.com/jobs/1",
                "external_job_id": "test-1"
            }], None)

        with (
            patch.object(self.app_module, "verify_company_and_job", return_value=self.verification_result()),
            patch.object(self.app_module.predictor, "predict", return_value=prediction),
            patch("backend.scheduler.crawl_career_vacancies", side_effect=crawl_site),
            ThreadPoolExecutor(max_workers=8) as executor
        ):
            futures = [
                executor.submit(submit_detection, index) for index in range(6)
            ]
            dashboard_future = executor.submit(read_dashboard)
            monitoring_future = executor.submit(monitor_site)
            statuses = [future.result(timeout=60) for future in futures]
            dashboard_status = dashboard_future.result(timeout=60)
            monitoring_result = monitoring_future.result(timeout=60)

        self.assertEqual(statuses, [200] * 6)
        self.assertEqual(dashboard_status, 200)
        self.assertTrue(monitoring_result["success"])
        self.assertEqual(monitoring_result["new_vacancies"], 1)

        connection = get_db_connection()
        saved = connection.execute(
            "SELECT id FROM job_submissions WHERE company_name LIKE ?",
            (f"Concurrent Test {marker}%",)
        ).fetchall()
        monitor_history = connection.execute(
            "SELECT COUNT(*) AS count FROM monitoring_history WHERE monitored_website_id = ?",
            (site_id,)
        ).fetchone()["count"]
        connection.close()
        self.assertEqual(len(saved), 6)
        self.assertEqual(monitor_history, 1)

        restarted_app = self.app_module.create_app()
        restarted_app.config.update(START_MONITORING_SCHEDULER=False)
        restart_response = restarted_app.test_client().get(
            f"/result/{saved[0]['id']}"
        )
        self.assertEqual(restart_response.status_code, 200)
        self.assertIn(b"Concurrent Test", restart_response.data)

    def test_sqlite_uses_wal_and_busy_timeout(self):
        from database import SQLITE_BUSY_TIMEOUT_MS, get_db_connection

        connection = get_db_connection()
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        connection.close()
        self.assertEqual(journal_mode.casefold(), "wal")
        self.assertEqual(busy_timeout, SQLITE_BUSY_TIMEOUT_MS)

    def test_writer_waits_for_existing_transaction_without_blocking_readers(self):
        from database import get_db_connection

        marker = uuid.uuid4().hex[:10]
        first_writer = get_db_connection()
        first_writer.execute("BEGIN IMMEDIATE")
        first_writer.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (f"writer-one-{marker}", f"writer-one-{marker}@example.com", "unused")
        )

        reader = get_db_connection()
        self.assertIsNotNone(
            reader.execute("SELECT COUNT(*) FROM users").fetchone()
        )
        reader.close()

        attempting_write = threading.Event()
        write_finished = threading.Event()
        write_errors = []

        def second_writer():
            connection = get_db_connection()
            try:
                attempting_write.set()
                connection.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (f"writer-two-{marker}", f"writer-two-{marker}@example.com", "unused")
                )
                connection.commit()
            except Exception as error:
                connection.rollback()
                write_errors.append(error)
            finally:
                connection.close()
                write_finished.set()

        writer_thread = threading.Thread(target=second_writer)
        writer_thread.start()
        self.assertTrue(attempting_write.wait(timeout=5))
        self.assertFalse(write_finished.wait(timeout=0.1))
        first_writer.commit()
        first_writer.close()
        writer_thread.join(timeout=10)

        self.assertFalse(writer_thread.is_alive())
        self.assertEqual(write_errors, [])
        self.assertTrue(write_finished.is_set())


if __name__ == "__main__":
    unittest.main()
