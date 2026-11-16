"""
Real-Time Fake Job Recruitment Detection System
Main Flask Application
"""

import os
import json
import functools
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, jsonify, abort
)
from werkzeug.security import generate_password_hash, check_password_hash

from config import Config
from database import get_db_connection, init_db
from backend.text_processor import clean_text
from backend.image_processor import validate_image_file, save_uploaded_image
from backend.ocr_engine import perform_ocr, is_ocr_available
from backend.ml_predictor import predictor
from backend.website_verifier import verify_company_and_job
from backend.job_matcher import calculate_job_match
from backend.scheduler import MonitoringScheduler
from backend.notification import NotificationService
from backend.assessment import (
    combine_recruitment_assessment,
    stored_official_verification_status
)


def create_app():
    """Application factory for the Flask app."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize SQLite database
    init_db()

    @app.before_request
    def ensure_monitoring_scheduler():
        if app.config.get("START_MONITORING_SCHEDULER", True):
            MonitoringScheduler.start_scheduler(Config.MONITORING_INTERVAL_HOURS)

    # -------------------------------------------------------------
    # Context Processors & Helpers
    # -------------------------------------------------------------
    def get_current_user():
        """Retrieve logged in user record from session."""
        user_id = session.get("user_id")
        if not user_id:
            return None
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()
        return dict(user) if user else None

    def login_required(view):
        """Decorator to require user authentication for sensitive views."""
        @functools.wraps(view)
        def wrapped_view(**kwargs):
            if session.get("user_id") is None:
                flash("Please log in to access this page.", "warning")
                return redirect(url_for("login", next=request.url))
            return view(**kwargs)
        return wrapped_view

    @app.context_processor
    def inject_global_data():
        """Inject user info and unread notification count into all templates."""
        user = get_current_user()
        unread_notifs = 0
        if user:
            conn = get_db_connection()
            row = conn.execute(
                "SELECT COUNT(*) as count FROM notifications WHERE user_id = ? AND is_read = 0",
                (user["id"],)
            ).fetchone()
            unread_notifs = row["count"] if row else 0
            conn.close()
        return {
            "current_user": user,
            "unread_notification_count": unread_notifs,
            "now_year": datetime.now().year
        }

    # -------------------------------------------------------------
    # Authentication Routes
    # -------------------------------------------------------------
    @app.route("/register", methods=["GET", "POST"])
    def register():
        """Handle user registration."""
        if session.get("user_id"):
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            full_name = request.form.get("full_name", "").strip()

            if not username or not email or not password:
                flash("Username, email, and password are required.", "danger")
                return render_template("register.html")

            if password != confirm_password:
                flash("Passwords do not match.", "danger")
                return render_template("register.html")

            if len(password) < 6:
                flash("Password must be at least 6 characters long.", "danger")
                return render_template("register.html")

            conn = get_db_connection()
            try:
                # Check for existing user
                existing = conn.execute(
                    "SELECT id FROM users WHERE username = ? OR email = ?",
                    (username, email)
                ).fetchone()
                if existing:
                    flash("Username or email already exists. Please choose another.", "danger")
                    conn.close()
                    return render_template("register.html")

                password_hash = generate_password_hash(password)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (username, email, password_hash, full_name)
                    VALUES (?, ?, ?, ?)
                """, (username, email, password_hash, full_name))
                user_id = cursor.lastrowid

                # Create default user preferences entry
                cursor.execute("""
                    INSERT INTO user_requirements (user_id, preferred_title, preferred_company, required_skills, preferred_location, minimum_salary, experience_level, job_type)
                    VALUES (?, '', '', '', '', 0, '', 'Full-time')
                """, (user_id,))

                conn.commit()
                conn.close()

                # Automatically log in the user
                session["user_id"] = user_id
                session["username"] = username
                flash("Registration successful! Welcome to the Fake Job Detection System.", "success")
                return redirect(url_for("dashboard"))
            except Exception as e:
                conn.close()
                flash(f"Registration error: {str(e)}", "danger")
                return render_template("register.html")

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        """Handle user login."""
        if session.get("user_id"):
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            login_id = request.form.get("login_id", "").strip()
            password = request.form.get("password", "")

            if not login_id or not password:
                flash("Please enter both username/email and password.", "danger")
                return render_template("login.html")

            conn = get_db_connection()
            user = conn.execute("""
                SELECT * FROM users WHERE username = ? OR email = ?
            """, (login_id, login_id.lower())).fetchone()
            conn.close()

            if user and check_password_hash(user["password_hash"], password):
                session.clear()
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                flash(f"Welcome back, {user['username']}!", "success")
                next_url = request.args.get("next")
                return redirect(next_url or url_for("dashboard"))
            else:
                flash("Invalid username/email or password.", "danger")

        return render_template("login.html")

    @app.route("/logout")
    def logout():
        """Log out the current user."""
        session.clear()
        flash("You have been securely logged out.", "info")
        return redirect(url_for("index"))

    # -------------------------------------------------------------
    # Core Application Routes
    # -------------------------------------------------------------
    @app.route("/")
    def index():
        """Public homepage displaying project overview and stats."""
        conn = get_db_connection()
        total_jobs = conn.execute("SELECT COUNT(*) as count FROM job_submissions").fetchone()["count"]
        likely_genuine = conn.execute("SELECT COUNT(*) as count FROM ml_predictions WHERE is_fake = 0").fetchone()["count"]
        likely_fake = conn.execute("SELECT COUNT(*) as count FROM ml_predictions WHERE is_fake = 1").fetchone()["count"]
        verification_rows = conn.execute("""
            SELECT verification_status, official_verification_status,
                   matching_vacancy_details
            FROM company_verification_results
        """).fetchall()
        verified_count = sum(
            stored_official_verification_status(dict(row)) == "VERIFIED GENUINE"
            for row in verification_rows
        )
        
        # Recent submissions sample
        recent_submissions = conn.execute("""
            SELECT s.id, s.company_name, s.job_title, s.created_at,
                   p.prediction_label, p.is_fake, p.confidence_score,
                   v.verification_status, v.official_verification_status,
                   v.matching_vacancy_details
            FROM job_submissions s
            LEFT JOIN ml_predictions p ON s.id = p.submission_id
            LEFT JOIN company_verification_results v ON s.id = v.submission_id
            ORDER BY s.created_at DESC LIMIT 5
        """).fetchall()
        conn.close()
        recent_submissions = [
            {
                **dict(row),
                "verification_status": stored_official_verification_status(dict(row))
            }
            for row in recent_submissions
        ]

        metrics = predictor.metrics
        return render_template(
            "index.html",
            total_jobs=total_jobs,
            likely_genuine=likely_genuine,
            likely_fake=likely_fake,
            verified_count=verified_count,
            recent_submissions=recent_submissions,
            metrics=metrics
        )

    @app.route("/dashboard")
    @login_required
    def dashboard():
        """User dashboard with statistics, alerts, and quick actions."""
        user_id = session["user_id"]
        conn = get_db_connection()

        # Summary counts
        total_analysed = conn.execute(
            "SELECT COUNT(*) as count FROM job_submissions WHERE user_id = ?", (user_id,)
        ).fetchone()["count"]

        genuine_count = conn.execute("""
            SELECT COUNT(*) as count FROM job_submissions s
            JOIN ml_predictions p ON s.id = p.submission_id
            WHERE s.user_id = ? AND p.is_fake = 0
        """, (user_id,)).fetchone()["count"]

        suspicious_count = conn.execute("""
            SELECT COUNT(*) as count FROM job_submissions s
            JOIN ml_predictions p ON s.id = p.submission_id
            WHERE s.user_id = ? AND p.is_fake = 1
        """, (user_id,)).fetchone()["count"]

        user_verification_rows = conn.execute("""
            SELECT v.verification_status, v.official_verification_status,
                   v.matching_vacancy_details
            FROM job_submissions s
            JOIN company_verification_results v ON s.id = v.submission_id
            WHERE s.user_id = ?
        """, (user_id,)).fetchall()
        unverified_count = sum(
            stored_official_verification_status(dict(row)) in {
                "UNVERIFIED",
                "UNVERIFIED - WEBSITE CHECK FAILED"
            }
            for row in user_verification_rows
        )

        monitored_companies = conn.execute(
            "SELECT COUNT(*) as count FROM monitored_websites WHERE user_id = ? AND status = 'active'", (user_id,)
        ).fetchone()["count"]

        # Recent history
        recent_jobs = conn.execute("""
            SELECT s.id, s.company_name, s.job_title, s.created_at, s.submission_type,
                   p.prediction_label, p.is_fake, p.confidence_score,
                   v.verification_status, v.official_verification_status,
                   v.matching_vacancy_details
            FROM job_submissions s
            LEFT JOIN ml_predictions p ON s.id = p.submission_id
            LEFT JOIN company_verification_results v ON s.id = v.submission_id
            WHERE s.user_id = ?
            ORDER BY s.created_at DESC LIMIT 6
        """, (user_id,)).fetchall()

        # Recent notifications
        recent_notifs = conn.execute("""
            SELECT * FROM notifications
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT 5
        """, (user_id,)).fetchall()

        # Monitored sites summary
        monitored_sites = conn.execute("""
            SELECT m.*, COUNT(e.id) as vacancy_count
            FROM monitored_websites m
            LEFT JOIN extracted_job_postings e ON m.id = e.monitored_website_id AND e.status = 'open'
            WHERE m.user_id = ?
            GROUP BY m.id
            ORDER BY m.created_at DESC
        """, (user_id,)).fetchall()

        conn.close()
        recent_jobs = [
            {
                **dict(row),
                "verification_status": stored_official_verification_status(dict(row))
            }
            for row in recent_jobs
        ]

        return render_template(
            "dashboard.html",
            total_analysed=total_analysed,
            genuine_count=genuine_count,
            suspicious_count=suspicious_count,
            unverified_count=unverified_count,
            monitored_companies=monitored_companies,
            recent_jobs=recent_jobs,
            recent_notifs=recent_notifs,
            monitored_sites=monitored_sites,
            metrics=predictor.metrics
        )

    # -------------------------------------------------------------
    # Detection Input Routes (Text & Image)
    # -------------------------------------------------------------
    @app.route("/detect/text", methods=["GET", "POST"])
    def text_detection():
        """Text-based job detection form and submission handler."""
        if request.method == "POST":
            company_name = request.form.get("company_name", "").strip()
            job_title = request.form.get("job_title", "").strip()
            job_description = request.form.get("job_description", "").strip()
            salary = request.form.get("salary", "").strip()
            job_location = request.form.get("job_location", "").strip()
            required_skills = request.form.get("required_skills", "").strip()
            experience_requirements = request.form.get("experience_requirements", "").strip()
            education = request.form.get("education", "").strip()
            recruiter_email = request.form.get("recruiter_email", "").strip()
            contact_number = request.form.get("contact_number", "").strip()
            application_url = request.form.get("application_url", "").strip()
            official_website_url = request.form.get("official_website_url", "").strip()
            official_careers_url = request.form.get("official_careers_url", "").strip()

            # Required field validation
            errors = []
            if not company_name:
                errors.append("Company Name is required.")
            if not job_title:
                errors.append("Job Title is required.")
            if not job_description:
                errors.append("Job Description is required.")

            if errors:
                for err in errors:
                    flash(err, "danger")
                return render_template("text_detection.html", form_data=request.form)

            user_id = session.get("user_id")

            # 1. Run Machine Learning Fake Job Prediction
            ml_result = predictor.predict(
                job_title=job_title,
                company_name=company_name,
                job_description=job_description,
                requirements=experience_requirements,
                skills=required_skills,
                salary=salary,
                location=job_location,
                recruiter_email=recruiter_email,
                application_url=application_url,
                education=education,
                contact_number=contact_number,
                official_website=official_website_url
            )

            # 2. Run Official Company Website Verification
            verif_result = verify_company_and_job(
                company_name=company_name,
                job_title=job_title,
                job_description=job_description,
                application_url=application_url,
                official_careers_url=official_careers_url,
                official_website_url=official_website_url,
                recruiter_email=recruiter_email,
                salary=salary,
                location=job_location
            )

            # 3. Save submission and results in SQLite
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO job_submissions (
                    user_id, company_name, job_title, job_description, salary,
                    job_location, required_skills, experience_requirements,
                    education, recruiter_email, contact_number, application_url,
                    official_website_url, official_careers_url, submission_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'text')
            """, (
                user_id, company_name, job_title, job_description, salary,
                job_location, required_skills, experience_requirements, education,
                recruiter_email, contact_number, application_url,
                official_website_url, official_careers_url
            ))
            submission_id = cursor.lastrowid

            # Save ML prediction
            cursor.execute("""
                INSERT INTO ml_predictions (
                    submission_id, prediction_label, is_fake, confidence_score,
                    fake_probability, suspicious_indicators, explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                submission_id,
                ml_result["prediction_label"],
                1 if ml_result["is_fake"] else 0,
                ml_result["confidence_score"],
                ml_result["fake_probability"],
                json.dumps(ml_result["suspicious_indicators"]),
                ml_result["explanation"]
            ))

            # Save Company Verification Result
            cursor.execute("""
                INSERT INTO company_verification_results (
                    submission_id, company_name, verification_status, official_verification_status,
                    official_website_url,
                    careers_page_url, vacancy_matched, matching_vacancy_details,
                    evidence, source_urls, explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                submission_id,
                company_name,
                verif_result["verification_status"],
                verif_result["official_verification_status"],
                verif_result["official_website_url"],
                verif_result["careers_page_url"],
                1 if verif_result["vacancy_matched"] else 0,
                json.dumps(verif_result["matching_vacancy_details"]) if verif_result["matching_vacancy_details"] else None,
                json.dumps(verif_result["evidence"]),
                json.dumps(verif_result["source_urls"]),
                verif_result["explanation"]
            ))

            # 4. If logged in, calculate Job Match with user preferences
            if user_id:
                user_req_row = cursor.execute("SELECT * FROM user_requirements WHERE user_id = ?", (user_id,)).fetchone()
                if user_req_row:
                    match_res = calculate_job_match(
                        user_req=dict(user_req_row),
                        job_title=job_title,
                        company_name=company_name,
                        job_description=job_description,
                        job_location=job_location,
                        salary=salary,
                        experience=experience_requirements,
                        job_skills=required_skills
                    )
                    cursor.execute("""
                        INSERT INTO job_match_results (
                            user_id, submission_id, match_score, title_match_score,
                            skills_match_score, location_match_score, experience_match_score,
                            salary_match_score, match_breakdown
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_id, submission_id, match_res["match_score"],
                        match_res["title_match_score"], match_res["skills_match_score"],
                        match_res["location_match_score"], match_res["experience_match_score"],
                        match_res["salary_match_score"], json.dumps(match_res["breakdown"])
                    ))

            conn.commit()
            conn.close()

            flash("Job analysis completed successfully!", "success")
            return redirect(url_for("result", submission_id=submission_id))

        return render_template("text_detection.html")

    @app.route("/detect/image", methods=["GET", "POST"])
    def image_detection():
        """Recruitment poster upload, OCR extraction, review & analysis."""
        return render_template("image_detection.html", ocr_available=is_ocr_available())

    @app.route("/api/ocr/extract", methods=["POST"])
    def api_ocr_extract():
        """AJAX endpoint for uploading image, previewing, and performing OCR text extraction."""
        if "poster_image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided in request."}), 400

        file = request.files["poster_image"]
        is_valid, err = validate_image_file(file)
        if not is_valid:
            return jsonify({"success": False, "error": err}), 400

        try:
            rel_url, abs_path, file_size = save_uploaded_image(file)
            ocr_res = perform_ocr(abs_path)

            return jsonify({
                "success": True,
                "image_url": rel_url,
                "file_path": abs_path,
                "file_name": os.path.basename(abs_path),
                "file_size": file_size,
                "ocr_success": ocr_res["success"],
                "raw_text": ocr_res["raw_text"],
                "ocr_engine": ocr_res["engine_used"],
                "ocr_error": ocr_res.get("error_message"),
                "extracted_fields": ocr_res["extracted_fields"]
            })
        except Exception as e:
            return jsonify({"success": False, "error": f"Image processing error: {str(e)}"}), 500

    @app.route("/detect/image/submit", methods=["POST"])
    def submit_image_detection():
        """Handle final submission of OCR-extracted & user-reviewed fields."""
        company_name = request.form.get("company_name", "").strip()
        job_title = request.form.get("job_title", "").strip()
        job_description = request.form.get("job_description", "").strip()
        salary = request.form.get("salary", "").strip()
        job_location = request.form.get("job_location", "").strip()
        required_skills = request.form.get("required_skills", "").strip()
        experience_requirements = request.form.get("experience_requirements", "").strip()
        education = request.form.get("education", "").strip()
        recruiter_email = request.form.get("recruiter_email", "").strip()
        contact_number = request.form.get("contact_number", "").strip()
        application_url = request.form.get("application_url", "").strip()
        official_website_url = request.form.get("official_website_url", "").strip()
        official_careers_url = request.form.get("official_careers_url", "").strip()

        image_path = request.form.get("image_path", "").strip()
        image_name = request.form.get("image_name", "").strip()
        raw_text = request.form.get("raw_text", "").strip()
        ocr_engine = request.form.get("ocr_engine", "Tesseract OCR").strip()

        errors = []
        if not company_name:
            errors.append("Company Name is required.")
        if not job_title:
            errors.append("Job Title is required.")
        if not job_description:
            errors.append("Job Description / Extracted text is required.")

        if errors:
            for err in errors:
                flash(err, "danger")
            return redirect(url_for("image_detection"))

        user_id = session.get("user_id")

        # 1. ML Prediction
        ml_result = predictor.predict(
            job_title=job_title,
            company_name=company_name,
            job_description=job_description,
            requirements=experience_requirements,
            skills=required_skills,
            salary=salary,
            location=job_location,
            recruiter_email=recruiter_email,
            application_url=application_url,
            education=education,
            contact_number=contact_number,
            official_website=official_website_url
        )

        # 2. Website Verification
        verif_result = verify_company_and_job(
            company_name=company_name,
            job_title=job_title,
            job_description=job_description,
            application_url=application_url,
            official_careers_url=official_careers_url,
            official_website_url=official_website_url,
            recruiter_email=recruiter_email,
            salary=salary,
            location=job_location
        )

        # 3. Store in Database
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO job_submissions (
                user_id, company_name, job_title, job_description, salary,
                job_location, required_skills, experience_requirements,
                education, recruiter_email, contact_number, application_url,
                official_website_url, official_careers_url, submission_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'poster')
        """, (
            user_id, company_name, job_title, job_description, salary,
            job_location, required_skills, experience_requirements, education,
            recruiter_email, contact_number, application_url,
            official_website_url, official_careers_url
        ))
        submission_id = cursor.lastrowid

        # Save poster upload metadata
        if image_path:
            file_sz = os.path.getsize(image_path) if os.path.exists(image_path) else 0
            cursor.execute("""
                INSERT INTO uploaded_posters (
                    submission_id, file_path, file_name, file_size, extracted_raw_text, ocr_engine_used
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (submission_id, image_path, image_name or os.path.basename(image_path), file_sz, raw_text, ocr_engine))

        # Save ML Prediction
        cursor.execute("""
            INSERT INTO ml_predictions (
                submission_id, prediction_label, is_fake, confidence_score,
                fake_probability, suspicious_indicators, explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            submission_id,
            ml_result["prediction_label"],
            1 if ml_result["is_fake"] else 0,
            ml_result["confidence_score"],
            ml_result["fake_probability"],
            json.dumps(ml_result["suspicious_indicators"]),
            ml_result["explanation"]
        ))

        # Save Verification Result
        cursor.execute("""
            INSERT INTO company_verification_results (
                submission_id, company_name, verification_status, official_verification_status,
                official_website_url,
                careers_page_url, vacancy_matched, matching_vacancy_details,
                evidence, source_urls, explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            submission_id,
            company_name,
            verif_result["verification_status"],
            verif_result["official_verification_status"],
            verif_result["official_website_url"],
            verif_result["careers_page_url"],
            1 if verif_result["vacancy_matched"] else 0,
            json.dumps(verif_result["matching_vacancy_details"]) if verif_result["matching_vacancy_details"] else None,
            json.dumps(verif_result["evidence"]),
            json.dumps(verif_result["source_urls"]),
            verif_result["explanation"]
        ))

        # 4. Job Match
        if user_id:
            user_req_row = cursor.execute("SELECT * FROM user_requirements WHERE user_id = ?", (user_id,)).fetchone()
            if user_req_row:
                match_res = calculate_job_match(
                    user_req=dict(user_req_row),
                    job_title=job_title,
                    company_name=company_name,
                    job_description=job_description,
                    job_location=job_location,
                    salary=salary,
                    experience=experience_requirements,
                    job_skills=required_skills
                )
                cursor.execute("""
                    INSERT INTO job_match_results (
                        user_id, submission_id, match_score, title_match_score,
                        skills_match_score, location_match_score, experience_match_score,
                        salary_match_score, match_breakdown
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, submission_id, match_res["match_score"],
                    match_res["title_match_score"], match_res["skills_match_score"],
                    match_res["location_match_score"], match_res["experience_match_score"],
                    match_res["salary_match_score"], json.dumps(match_res["breakdown"])
                ))

        conn.commit()
        conn.close()

        flash("Recruitment poster processed and analysed successfully!", "success")
        return redirect(url_for("result", submission_id=submission_id))

    # -------------------------------------------------------------
    # Analysis Result Page (Section 8)
    # -------------------------------------------------------------
    @app.route("/result/<int:submission_id>")
    def result(submission_id: int):
        """Display the complete, structured analysis result page."""
        conn = get_db_connection()
        submission = conn.execute("SELECT * FROM job_submissions WHERE id = ?", (submission_id,)).fetchone()
        if not submission:
            conn.close()
            abort(404, description="Job submission not found.")
        if submission["user_id"] is not None and submission["user_id"] != session.get("user_id"):
            conn.close()
            abort(404, description="Job submission not found.")

        ml_pred = conn.execute("SELECT * FROM ml_predictions WHERE submission_id = ?", (submission_id,)).fetchone()
        verif = conn.execute("SELECT * FROM company_verification_results WHERE submission_id = ?", (submission_id,)).fetchone()
        poster = conn.execute("SELECT * FROM uploaded_posters WHERE submission_id = ?", (submission_id,)).fetchone()
        match_row = conn.execute("SELECT * FROM job_match_results WHERE submission_id = ?", (submission_id,)).fetchone()
        conn.close()

        # Parse JSON fields safely
        suspicious_indicators = []
        if ml_pred and ml_pred["suspicious_indicators"]:
            try:
                suspicious_indicators = json.loads(ml_pred["suspicious_indicators"])
            except Exception:
                suspicious_indicators = []

        evidence = []
        source_urls = []
        matching_vacancy = None
        if verif:
            try:
                evidence = json.loads(verif["evidence"]) if verif["evidence"] else []
            except Exception:
                evidence = []
            try:
                source_urls = json.loads(verif["source_urls"]) if verif["source_urls"] else []
            except Exception:
                source_urls = []
            try:
                matching_vacancy = json.loads(verif["matching_vacancy_details"]) if verif["matching_vacancy_details"] else None
            except Exception:
                matching_vacancy = None

        ml_data = dict(ml_pred) if ml_pred else None
        verif_data = dict(verif) if verif else None
        if verif_data:
            official_status = stored_official_verification_status(
                verif_data, matching_vacancy
            )
            has_specific_listing_evidence = official_status == "VERIFIED GENUINE"
            if not has_specific_listing_evidence:
                matching_vacancy = None
                verif_data["vacancy_matched"] = 0
                legacy_verified_claim = (
                    verif_data.get("verification_status") == "VERIFIED GENUINE"
                    or verif_data.get("official_verification_status") == "Verified on Official Website"
                )
                if legacy_verified_claim:
                    evidence = []
                    verif_data["explanation"] = (
                        "This legacy result did not record direct published job-listing evidence. "
                        "The vacancy is unverified; this does not prove that it is fake."
                    )
            verif_data["official_verification_status"] = official_status
        else:
            official_status = "UNVERIFIED"

        final_status, final_reason = combine_recruitment_assessment(ml_data, official_status)

        match_breakdown = {}
        if match_row and match_row["match_breakdown"]:
            try:
                match_breakdown = json.loads(match_row["match_breakdown"])
            except Exception:
                match_breakdown = {}

        return render_template(
            "result.html",
            submission=dict(submission),
            ml=ml_data,
            verif=verif_data,
            poster=dict(poster) if poster else None,
            match=dict(match_row) if match_row else None,
            suspicious_indicators=suspicious_indicators,
            evidence=evidence,
            source_urls=source_urls,
            matching_vacancy=matching_vacancy,
            official_verification_status=official_status,
            final_status=final_status,
            final_reason=final_reason,
            match_breakdown=match_breakdown,
            metrics=predictor.metrics
        )

    # -------------------------------------------------------------
    # Standalone Official Company Verification Page
    # -------------------------------------------------------------
    @app.route("/verify", methods=["GET", "POST"])
    def verification():
        """Standalone company verification page."""
        result_data = None
        if request.method == "POST":
            session.pop("verification_result", None)
            company_name = request.form.get("company_name", "").strip()
            job_title = request.form.get("job_title", "").strip()
            careers_url = request.form.get("careers_url", "").strip()
            app_url = request.form.get("app_url", "").strip()
            recruiter_email = request.form.get("recruiter_email", "").strip()

            if not company_name or not job_title:
                flash("Company Name and Job Title are required for official verification.", "danger")
            else:
                result_data = verify_company_and_job(
                    company_name=company_name,
                    job_title=job_title,
                    job_description="Verification search query",
                    application_url=app_url,
                    official_careers_url=careers_url,
                    recruiter_email=recruiter_email
                )

        response = app.make_response(
            render_template("verification.html", result=result_data)
        )
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    # -------------------------------------------------------------
    # Job Monitoring Routes (Section 10)
    # -------------------------------------------------------------
    @app.route("/monitoring", methods=["GET", "POST"])
    @login_required
    def monitoring():
        """Job monitoring page: manage monitored companies and view discovered vacancies."""
        user_id = session["user_id"]
        conn = get_db_connection()

        if request.method == "POST":
            company_name = request.form.get("company_name", "").strip()
            careers_url = request.form.get("careers_url", "").strip()

            if not company_name or not careers_url:
                flash("Both Company Name and Careers URL are required.", "danger")
            else:
                is_safe, reason = verify_company_and_job.__globals__["is_safe_url"](careers_url)
                if not is_safe:
                    flash(f"URL rejected by security policy: {reason}", "danger")
                else:
                    try:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO monitored_websites (user_id, company_name, careers_url, status, last_status_message)
                            VALUES (?, ?, ?, 'active', 'Monitoring scheduled. Initial check pending.')
                        """, (user_id, company_name, careers_url))
                        site_id = cursor.lastrowid
                        conn.commit()

                        # Perform immediate initial check
                        MonitoringScheduler.check_single_website(site_id, user_id, company_name, careers_url)
                        flash(f"Successfully added '{company_name}' to 24-hour monitoring!", "success")
                    except Exception as e:
                        if "UNIQUE constraint failed" in str(e):
                            flash("You are already monitoring this company and URL.", "warning")
                        else:
                            flash(f"Error adding monitoring entry: {str(e)}", "danger")

        # Fetch all monitored sites for user
        monitored_sites = conn.execute("""
            SELECT m.*, COUNT(e.id) as open_vacancies_count
            FROM monitored_websites m
            LEFT JOIN extracted_job_postings e ON m.id = e.monitored_website_id AND e.status = 'open'
            WHERE m.user_id = ?
            GROUP BY m.id
            ORDER BY m.created_at DESC
        """, (user_id,)).fetchall()

        # Fetch discovered job postings
        postings = conn.execute("""
            SELECT e.*, m.company_name
            FROM extracted_job_postings e
            JOIN monitored_websites m ON e.monitored_website_id = m.id
            WHERE m.user_id = ?
            ORDER BY e.last_seen_at DESC LIMIT 50
        """, (user_id,)).fetchall()

        # Fetch monitoring history
        history = conn.execute("""
            SELECT h.*, m.company_name
            FROM monitoring_history h
            JOIN monitored_websites m ON h.monitored_website_id = m.id
            WHERE m.user_id = ?
            ORDER BY h.check_timestamp DESC LIMIT 20
        """, (user_id,)).fetchall()

        conn.close()

        return render_template(
            "monitoring.html",
            monitored_sites=monitored_sites,
            postings=postings,
            history=history
        )

    @app.route("/monitoring/action/<int:site_id>/<string:action>")
    @login_required
    def monitoring_action(site_id: int, action: str):
        """Pause, resume, delete, or run immediate check for a monitored site."""
        user_id = session["user_id"]
        conn = get_db_connection()
        site = conn.execute(
            "SELECT * FROM monitored_websites WHERE id = ? AND user_id = ?",
            (site_id, user_id)
        ).fetchone()

        if not site:
            conn.close()
            flash("Monitored website record not found.", "danger")
            return redirect(url_for("monitoring"))

        cursor = conn.cursor()
        if action == "pause":
            cursor.execute("UPDATE monitored_websites SET status = 'paused' WHERE id = ?", (site_id,))
            conn.commit()
            flash(f"Monitoring paused for '{site['company_name']}'.", "info")
        elif action == "resume":
            cursor.execute("UPDATE monitored_websites SET status = 'active' WHERE id = ?", (site_id,))
            conn.commit()
            flash(f"Monitoring resumed for '{site['company_name']}'.", "success")
        elif action == "delete":
            cursor.execute("DELETE FROM monitored_websites WHERE id = ?", (site_id,))
            conn.commit()
            flash(f"Removed '{site['company_name']}' from monitoring list.", "info")
        elif action == "check_now":
            conn.close()
            res = MonitoringScheduler.check_single_website(
                site_id, user_id, site["company_name"], site["careers_url"]
            )
            flash(f"Check completed: {res.get('status_message', 'Done')}", "info")
            return redirect(url_for("monitoring"))
        else:
            flash("Unknown action.", "danger")

        if action != "check_now":
            conn.close()

        return redirect(url_for("monitoring"))

    # -------------------------------------------------------------
    # Notifications Route (Section 11)
    # -------------------------------------------------------------
    @app.route("/notifications")
    @login_required
    def notifications():
        """User notifications page."""
        user_id = session["user_id"]
        conn = get_db_connection()
        notifs = conn.execute("""
            SELECT * FROM notifications
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (user_id,)).fetchall()
        conn.close()
        return render_template("notifications.html", notifications=notifs)

    @app.route("/notifications/mark_read/<int:notif_id>")
    @login_required
    def mark_notification_read(notif_id: int):
        """Mark single notification as read."""
        user_id = session["user_id"]
        conn = get_db_connection()
        conn.execute("UPDATE notifications SET is_read = 1 WHERE id = ? AND user_id = ?", (notif_id, user_id))
        conn.commit()
        conn.close()
        return redirect(url_for("notifications"))

    @app.route("/notifications/mark_all_read")
    @login_required
    def mark_all_notifications_read():
        """Mark all notifications for user as read."""
        user_id = session["user_id"]
        conn = get_db_connection()
        conn.execute("UPDATE notifications SET is_read = 1 WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        flash("All notifications marked as read.", "success")
        return redirect(url_for("notifications"))

    # -------------------------------------------------------------
    # Detection History Page
    # -------------------------------------------------------------
    @app.route("/history")
    @login_required
    def history():
        """Detection history page listing all previously submitted jobs."""
        user_id = session["user_id"]
        conn = get_db_connection()
        submissions = conn.execute("""
            SELECT s.id, s.company_name, s.job_title, s.salary, s.job_location,
                   s.submission_type, s.created_at,
                   p.prediction_label, p.is_fake, p.confidence_score,
                   v.verification_status, v.official_verification_status,
                   v.matching_vacancy_details, v.official_website_url
            FROM job_submissions s
            LEFT JOIN ml_predictions p ON s.id = p.submission_id
            LEFT JOIN company_verification_results v ON s.id = v.submission_id
            WHERE s.user_id = ?
            ORDER BY s.created_at DESC
        """, (user_id,)).fetchall()
        conn.close()
        submissions = [
            {
                **dict(row),
                "verification_status": stored_official_verification_status(dict(row))
            }
            for row in submissions
        ]
        return render_template("history.html", submissions=submissions)

    # -------------------------------------------------------------
    # User Profile / Requirements Page (Section 9)
    # -------------------------------------------------------------
    @app.route("/profile", methods=["GET", "POST"])
    @login_required
    def profile():
        """Manage user job preferences for matching & view ML model information."""
        user_id = session["user_id"]
        conn = get_db_connection()

        if request.method == "POST":
            preferred_title = request.form.get("preferred_title", "").strip()
            preferred_company = request.form.get("preferred_company", "").strip()
            required_skills = request.form.get("required_skills", "").strip()
            preferred_location = request.form.get("preferred_location", "").strip()
            minimum_salary = float(request.form.get("minimum_salary") or 0.0)
            experience_level = request.form.get("experience_level", "").strip()
            job_type = request.form.get("job_type", "Full-time").strip()

            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_requirements (
                    user_id, preferred_title, preferred_company, required_skills,
                    preferred_location, minimum_salary, experience_level, job_type, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    preferred_title = excluded.preferred_title,
                    preferred_company = excluded.preferred_company,
                    required_skills = excluded.required_skills,
                    preferred_location = excluded.preferred_location,
                    minimum_salary = excluded.minimum_salary,
                    experience_level = excluded.experience_level,
                    job_type = excluded.job_type,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                user_id, preferred_title, preferred_company, required_skills,
                preferred_location, minimum_salary, experience_level, job_type
            ))
            conn.commit()
            flash("Job preferences updated successfully!", "success")

        user_req = conn.execute("SELECT * FROM user_requirements WHERE user_id = ?", (user_id,)).fetchone()
        user_info = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()

        return render_template(
            "profile.html",
            user_req=dict(user_req) if user_req else {},
            user_info=dict(user_info) if user_info else {},
            metrics=predictor.metrics
        )

    # -------------------------------------------------------------
    # Error Handlers
    # -------------------------------------------------------------
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("index.html", error_message="404: The requested page could not be found."), 404

    @app.errorhandler(413)
    def request_entity_too_large(e):
        flash("Uploaded file exceeds the maximum allowed file size (16 MB).", "danger")
        return redirect(url_for("image_detection")), 413

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template("index.html", error_message="500: An unexpected internal error occurred."), 500

    return app


app = create_app()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=False)
