"""
Test suite validating the UI redesign:
- Verifies that all redesigned templates render cleanly with HTTP 200.
- Verifies that glassmorphism, cooper highlight classes, and buttons are present.
- Verifies that forms, navigation links, and static assets load successfully.
"""

import unittest
from app import create_app
from database import get_db_connection


class TestUIRedesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.client = cls.app.test_client()

    def test_homepage_rendering(self):
        """Test homepage renders with hero, Cooper highlights, and SVG illustration."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Fake Job Detection", html)
        self.assertIn("cooper-highlight", html)
        self.assertIn("hero-security-ai.svg", html)
        self.assertIn("glass-card", html)
        self.assertIn("btn-primary-gradient", html)

    def test_text_detection_page(self):
        """Test text detection page contains modern inputs and counters."""
        resp = self.client.get("/detect/text")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("name=\"company_name\"", html)
        self.assertIn("name=\"job_title\"", html)
        self.assertIn("name=\"job_description\"", html)
        self.assertIn("desc_char_count", html)
        self.assertIn("btn-primary-gradient", html)

    def test_image_detection_page(self):
        """Test image detection page contains dropzone and editable fields."""
        resp = self.client.get("/detect/image")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("poster-dropzone", html)
        self.assertIn("dropzone-container", html)
        self.assertIn("field_company_name", html)

    def test_verification_page(self):
        """Test standalone company verification page renders cleanly."""
        resp = self.client.get("/verify")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Official Company Website Verification", html)
        self.assertIn("name=\"company_name\"", html)
        self.assertIn("btn-primary-gradient", html)

    def test_login_and_register_pages(self):
        """Test authentication pages render glass cards."""
        resp_login = self.client.get("/login")
        self.assertEqual(resp_login.status_code, 200)
        html_login = resp_login.get_data(as_text=True)
        self.assertIn("glass-card", html_login)
        self.assertIn("Sign In to Dashboard", html_login)

        resp_reg = self.client.get("/register")
        self.assertEqual(resp_reg.status_code, 200)
        html_reg = resp_reg.get_data(as_text=True)
        self.assertIn("glass-card", html_reg)
        self.assertIn("Create Free Account", html_reg)

    def test_static_assets(self):
        """Test that redesigned CSS and SVGs are accessible."""
        resp_css = self.client.get("/static/css/style.css")
        self.assertEqual(resp_css.status_code, 200)
        css_text = resp_css.get_data(as_text=True)
        self.assertIn("--font-cooper", css_text)
        self.assertIn("backdrop-filter: blur(16px)", css_text)
        self.assertIn("btn-primary-gradient", css_text)

        resp_svg = self.client.get("/static/images/hero-security-ai.svg")
        self.assertEqual(resp_svg.status_code, 200)


if __name__ == "__main__":
    unittest.main()
