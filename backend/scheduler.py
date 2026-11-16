"""
Background Job Monitoring Scheduler.
Uses APScheduler to periodically inspect monitored company career pages, detect new/reopened
vacancies, perform match calculations, and trigger notifications.
"""

import errno
import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import Config
from database import get_db_connection
from backend.job_extractor import crawl_career_vacancies
from backend.job_matcher import calculate_job_match
from backend.notification import NotificationService
from backend.website_verifier import is_authorized_recruitment_portal


class MonitoringScheduler:
    """Manages periodic 24-hour job monitoring across user-monitored company career sites."""

    _scheduler: Optional[BackgroundScheduler] = None
    _scheduler_lock_file = None
    _start_lock = threading.Lock()

    @classmethod
    def get_scheduler(cls) -> BackgroundScheduler:
        """Get or initialize the BackgroundScheduler singleton."""
        if cls._scheduler is None:
            cls._scheduler = BackgroundScheduler(daemon=True)
        return cls._scheduler

    @classmethod
    def start_scheduler(cls, interval_hours: Optional[int] = None):
        """Start monitoring in only one process for this database."""
        with cls._start_lock:
            scheduler = cls.get_scheduler()
            if scheduler.running:
                return True
            if not cls._acquire_scheduler_lock():
                print("APScheduler is already active in another application process.")
                return False

            hours = interval_hours or Config.MONITORING_INTERVAL_HOURS
            try:
                scheduler.add_job(
                    func=cls.run_monitoring_cycle,
                    trigger=IntervalTrigger(hours=hours),
                    id="periodic_job_monitor",
                    name="24-Hour Job Vacancy Monitor",
                    replace_existing=True
                )
                scheduler.start()
            except Exception:
                cls._release_scheduler_lock()
                raise
            print(f"APScheduler started: Monitoring cycle runs every {hours} hour(s).")
            return True

    @classmethod
    def _acquire_scheduler_lock(cls) -> bool:
        """Acquire a cross-process file lock tied to the active database."""
        lock_path = os.path.abspath(Config.DATABASE_PATH) + ".scheduler.lock"
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        lock_file = open(lock_path, "a+b")
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)

        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            lock_file.close()
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return False
            raise

        cls._scheduler_lock_file = lock_file
        return True

    @classmethod
    def _release_scheduler_lock(cls):
        """Release the monitoring process lock when its scheduler stops."""
        lock_file = cls._scheduler_lock_file
        if not lock_file:
            return
        try:
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
            cls._scheduler_lock_file = None

    @classmethod
    def shutdown_scheduler(cls):
        """Cleanly shut down the background scheduler."""
        with cls._start_lock:
            try:
                if cls._scheduler and cls._scheduler.running:
                    cls._scheduler.shutdown(wait=False)
                    print("APScheduler shut down.")
            finally:
                cls._release_scheduler_lock()

    @classmethod
    def run_monitoring_cycle(cls) -> Dict[str, Any]:
        """
        Execute a full monitoring run across all active monitored websites.
        Can be triggered automatically or manually from the dashboard.
        """
        conn = get_db_connection()
        try:
            active_sites = conn.execute(
                "SELECT * FROM monitored_websites WHERE status = 'active'"
            ).fetchall()
        finally:
            conn.close()

        results = {
            "monitored_sites_checked": len(active_sites),
            "total_new_vacancies": 0,
            "total_reopened_vacancies": 0,
            "total_alerts_dispatched": 0,
            "timestamp": datetime.now().isoformat()
        }

        for site in active_sites:
            site_id = site["id"]
            user_id = site["user_id"]
            company_name = site["company_name"]
            careers_url = site["careers_url"]

            try:
                site_res = cls.check_single_website(site_id, user_id, company_name, careers_url)
                results["total_new_vacancies"] += site_res.get("new_vacancies", 0)
                results["total_reopened_vacancies"] += site_res.get("reopened_vacancies", 0)
                results["total_alerts_dispatched"] += site_res.get("alerts_dispatched", 0)
            except Exception as e:
                print(f"Error checking monitored website ID {site_id} ({company_name}): {e}")

        return results

    @classmethod
    def check_single_website(
        cls,
        site_id: int,
        user_id: int,
        company_name: str,
        careers_url: str
    ) -> Dict[str, Any]:
        """Inspect a single monitored website, detect vacancy changes, and trigger alerts."""
        conn = get_db_connection()

        # Read user data and close before making a potentially slow network request.
        try:
            user_row = conn.execute(
                "SELECT email FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            user_email = user_row["email"] if user_row else ""
            user_req_row = conn.execute(
                "SELECT * FROM user_requirements WHERE user_id = ?", (user_id,)
            ).fetchone()
            user_req = dict(user_req_row) if user_req_row else {}
        finally:
            conn.close()

        discovered_jobs, error = crawl_career_vacancies(careers_url)
        now_ts = datetime.now()

        if error:
            conn = get_db_connection()
            try:
                conn.execute("""
                    INSERT INTO monitoring_history (
                        monitored_website_id, vacancies_found, new_vacancies_found,
                        reopened_vacancies_found, status, notes
                    ) VALUES (?, 0, 0, 0, 'FAILURE', ?)
                """, (site_id, f"Fetch failed: {error}"))
                conn.execute("""
                    UPDATE monitored_websites
                    SET last_checked_at = ?, last_status_message = ?
                    WHERE id = ?
                """, (now_ts, f"Check failed: {error}", site_id))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
            return {
                "success": False,
                "error": error,
                "new_vacancies": 0,
                "reopened_vacancies": 0,
                "alerts_dispatched": 0
            }

        conn = get_db_connection()
        try:
            existing_postings = {
                row["job_title"].lower(): dict(row)
                for row in conn.execute(
                    "SELECT * FROM extracted_job_postings WHERE monitored_website_id = ?",
                    (site_id,)
                ).fetchall()
            }
        finally:
            conn.close()

        new_count = 0
        reopened_count = 0
        seen_titles = set()
        prepared_jobs = []

        for job in discovered_jobs:
            title = job["job_title"]
            title_key = title.lower()
            seen_titles.add(title_key)
            job_url = job.get("job_url", careers_url)
            loc = job.get("job_location", "")
            dept = job.get("department", "")
            sal = job.get("salary", "")
            desc = job.get("description", "")
            ext_id = job.get("external_job_id", "")

            existing = existing_postings.get(title_key)
            is_new = existing is None
            is_reopened = bool(existing and existing["status"] == "closed")
            if is_new:
                new_count += 1
            elif is_reopened:
                reopened_count += 1

            match_res = None
            if (is_new or is_reopened) and user_req:
                match_res = calculate_job_match(
                    user_req=user_req,
                    job_title=title,
                    company_name=company_name,
                    job_description=desc,
                    job_location=loc,
                    salary=sal
                )
            prepared_jobs.append({
                "job": job,
                "existing": existing,
                "is_reopened": is_reopened,
                "match": match_res
            })

        return cls._persist_scan_results(
            site_id=site_id,
            user_id=user_id,
            user_email=user_email,
            company_name=company_name,
            careers_url=careers_url,
            discovered_jobs=discovered_jobs,
            existing_postings=existing_postings,
            seen_titles=seen_titles,
            prepared_jobs=prepared_jobs,
            new_count=new_count,
            reopened_count=reopened_count,
            now_ts=now_ts
        )

    @classmethod
    def _persist_scan_results(
        cls,
        site_id: int,
        user_id: int,
        user_email: str,
        company_name: str,
        careers_url: str,
        discovered_jobs: List[Dict[str, Any]],
        existing_postings: Dict[str, Dict[str, Any]],
        seen_titles: set,
        prepared_jobs: List[Dict[str, Any]],
        new_count: int,
        reopened_count: int,
        now_ts: datetime
    ) -> Dict[str, Any]:
        """Persist a scan atomically, then dispatch notifications outside the transaction."""
        conn = get_db_connection()
        pending_alerts = []
        try:
            cursor = conn.cursor()
            for item in prepared_jobs:
                job = item["job"]
                title = job["job_title"]
                job_url = job.get("job_url", careers_url)
                loc = job.get("job_location", "")
                dept = job.get("department", "")
                sal = job.get("salary", "")
                desc = job.get("description", "")
                ext_id = job.get("external_job_id", "")
                existing = item["existing"]

                if existing:
                    posting_id = existing["id"]
                    cursor.execute("""
                        UPDATE extracted_job_postings
                        SET status = 'open', last_seen_at = ?, job_url = ?
                        WHERE id = ?
                    """, (now_ts, job_url, posting_id))
                else:
                    cursor.execute("""
                        INSERT INTO extracted_job_postings (
                            monitored_website_id, job_title, job_location, department,
                            salary, description, job_url, external_job_id, status,
                            first_seen_at, last_seen_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
                    """, (
                        site_id, title, loc, dept, sal, desc, job_url, ext_id,
                        now_ts, now_ts
                    ))
                    posting_id = cursor.lastrowid

                match_res = item["match"]
                if match_res:
                    score = match_res["match_score"]
                    cursor.execute("""
                        INSERT INTO job_match_results (
                            user_id, posting_id, match_score, title_match_score,
                            skills_match_score, location_match_score, experience_match_score,
                            salary_match_score, match_breakdown
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_id, posting_id, score, match_res["title_match_score"],
                        match_res["skills_match_score"], match_res["location_match_score"],
                        match_res["experience_match_score"], match_res["salary_match_score"],
                        json.dumps(match_res["breakdown"])
                    ))

                    if match_res["qualifies_for_alert"]:
                        status_note = (
                            "Reopened Vacancy" if item["is_reopened"] else "New Vacancy"
                        )
                        pending_alerts.append({
                            "user_id": user_id,
                            "recipient_email": user_email,
                            "company_name": company_name,
                            "job_title": title,
                            "match_percentage": score,
                            "verification_status": "VERIFIED GENUINE",
                            "source_link": job_url,
                            "application_url": job_url,
                            "location": loc,
                            "salary": sal,
                            "custom_message": (
                                f"[{status_note}] {title} at {company_name} matches "
                                f"your profile with a score of {score}%."
                            )
                        })

            for title_key, existing in existing_postings.items():
                if title_key not in seen_titles and existing["status"] == "open":
                    cursor.execute(
                        "UPDATE extracted_job_postings SET status = 'closed' WHERE id = ?",
                        (existing["id"],)
                    )

            note = (
                f"Scan completed: {len(discovered_jobs)} active vacancies found "
                f"({new_count} new, {reopened_count} reopened, "
                f"{len(pending_alerts)} high-match alerts)."
            )
            cursor.execute("""
                INSERT INTO monitoring_history (
                    monitored_website_id, vacancies_found, new_vacancies_found,
                    reopened_vacancies_found, status, notes
                ) VALUES (?, ?, ?, ?, 'SUCCESS', ?)
            """, (site_id, len(discovered_jobs), new_count, reopened_count, note))
            cursor.execute("""
                UPDATE monitored_websites
                SET last_checked_at = ?, last_status_message = ?
                WHERE id = ?
            """, (now_ts, note, site_id))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        for alert in pending_alerts:
            NotificationService.dispatch_alert(**alert)

        return {
            "success": True,
            "vacancies_found": len(discovered_jobs),
            "new_vacancies": new_count,
            "reopened_vacancies": reopened_count,
            "alerts_dispatched": len(pending_alerts),
            "status_message": note
        }
