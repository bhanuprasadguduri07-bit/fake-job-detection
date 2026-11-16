"""
Notification service for Fake Job Recruitment Detection.
Handles in-app dashboard notifications and SMTP email dispatch with error resilience.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List, Tuple
from config import Config
from database import get_db_connection


class NotificationService:
    """Manages dashboard notifications and SMTP email alerts."""

    @staticmethod
    def create_dashboard_notification(
        user_id: int,
        title: str,
        company_name: str,
        job_title: str,
        match_percentage: float,
        verification_status: str,
        source_link: str,
        message: str,
        email_sent: bool = False
    ) -> int:
        """Create a new notification in the SQLite database."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO notifications (
                user_id, title, company_name, job_title, match_percentage,
                verification_status, source_link, message, is_read, email_sent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """, (
            user_id, title, company_name, job_title, match_percentage,
            verification_status, source_link, message, 1 if email_sent else 0
        ))
        notif_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return notif_id

    @staticmethod
    def is_smtp_configured() -> bool:
        """Check if SMTP credentials are provided in configuration."""
        return bool(Config.SMTP_SERVER and Config.SMTP_USERNAME and Config.SMTP_PASSWORD and Config.SMTP_SENDER_EMAIL)

    @classmethod
    def send_email_notification(
        cls,
        recipient_email: str,
        company_name: str,
        job_title: str,
        match_percentage: float,
        verification_status: str,
        source_link: str,
        application_url: Optional[str] = "",
        location: Optional[str] = "",
        salary: Optional[str] = ""
    ) -> Tuple[bool, Optional[str]]:
        """
        Send an email notification via SMTP.
        Returns (success, error_message).
        """
        if not cls.is_smtp_configured():
            return False, "SMTP is not configured in .env (SMTP_USERNAME / SMTP_PASSWORD missing)."

        if not recipient_email or "@" not in recipient_email:
            return False, "Invalid recipient email address."

        subject = f"[Job Alert] New Match ({match_percentage}%): {job_title} at {company_name}"

        # Compose plain text and HTML bodies
        text_body = f"""
New Vacancy Alert:
----------------------------------------
Company: {company_name}
Job Title: {job_title}
Match Score: {match_percentage}%
Verification Status: {verification_status}
Location: {location or 'Not specified'}
Salary: {salary or 'Not specified'}
Official Source: {source_link}
Application URL: {application_url or source_link}
----------------------------------------
Verified by Fake Job Recruitment Detection System
"""

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; }}
    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e1e4e8; border-radius: 8px; }}
    .header {{ background: #0d6efd; color: #fff; padding: 15px 20px; border-radius: 6px 6px 0 0; }}
    .content {{ padding: 20px; background: #fff; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 13px; }}
    .badge-verified {{ background: #d1e7dd; color: #0f5132; }}
    .badge-suspicious {{ background: #f8d7da; color: #842029; }}
    .badge-unverified {{ background: #fff3cd; color: #664d03; }}
    .btn {{ display: inline-block; padding: 10px 18px; background: #0d6efd; color: #fff; text-decoration: none; border-radius: 4px; font-weight: bold; margin-top: 15px; }}
    .footer {{ font-size: 12px; color: #6c757d; margin-top: 20px; text-align: center; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h2 style="margin:0;">Job Match Alert ({match_percentage}%)</h2>
    </div>
    <div class="content">
      <p>A new recruitment vacancy matching your career preferences has been identified.</p>
      <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
        <tr><td style="padding: 6px 0; color: #6c757d;"><strong>Company:</strong></td><td>{company_name}</td></tr>
        <tr><td style="padding: 6px 0; color: #6c757d;"><strong>Job Title:</strong></td><td><strong>{job_title}</strong></td></tr>
        <tr><td style="padding: 6px 0; color: #6c757d;"><strong>Match Score:</strong></td><td><strong style="color: #0d6efd;">{match_percentage}%</strong></td></tr>
        <tr><td style="padding: 6px 0; color: #6c757d;"><strong>Verification Status:</strong></td>
            <td><span class="badge { 'badge-verified' if verification_status == 'VERIFIED GENUINE' else ('badge-suspicious' if verification_status == 'LIKELY SUSPICIOUS' else 'badge-unverified') }">{verification_status}</span></td></tr>
        <tr><td style="padding: 6px 0; color: #6c757d;"><strong>Location:</strong></td><td>{location or 'Not specified'}</td></tr>
        <tr><td style="padding: 6px 0; color: #6c757d;"><strong>Salary:</strong></td><td>{salary or 'Not specified'}</td></tr>
      </table>
      <div style="margin-top: 20px;">
        <a href="{source_link}" class="btn" style="color: #ffffff;">View Official Source</a>
      </div>
    </div>
    <div class="footer">
      <p>This automated notification was generated by the Real-Time Fake Job Recruitment Detection System.</p>
    </div>
  </div>
</body>
</html>
"""

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = Config.SMTP_SENDER_EMAIL
            msg["To"] = recipient_email

            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            server = smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT, timeout=10)
            if Config.SMTP_USE_TLS:
                server.starttls()
            server.login(Config.SMTP_USERNAME, Config.SMTP_PASSWORD)
            server.sendmail(Config.SMTP_SENDER_EMAIL, [recipient_email], msg.as_string())
            server.quit()

            return True, None
        except Exception as e:
            return False, f"SMTP delivery failed: {str(e)}"

    @classmethod
    def dispatch_alert(
        cls,
        user_id: int,
        recipient_email: str,
        company_name: str,
        job_title: str,
        match_percentage: float,
        verification_status: str,
        source_link: str,
        application_url: Optional[str] = "",
        location: Optional[str] = "",
        salary: Optional[str] = "",
        custom_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Dispatches both dashboard notification and email (if configured).
        Preserves dashboard notification even if email sending fails.
        """
        msg_text = custom_message or (
            f"Matching vacancy '{job_title}' discovered at {company_name} "
            f"with {match_percentage}% match score (Status: {verification_status})."
        )

        email_ok, email_err = cls.send_email_notification(
            recipient_email=recipient_email,
            company_name=company_name,
            job_title=job_title,
            match_percentage=match_percentage,
            verification_status=verification_status,
            source_link=source_link,
            application_url=application_url,
            location=location,
            salary=salary
        )

        notif_id = cls.create_dashboard_notification(
            user_id=user_id,
            title=f"New Vacancy Match ({match_percentage}%): {job_title}",
            company_name=company_name,
            job_title=job_title,
            match_percentage=match_percentage,
            verification_status=verification_status,
            source_link=source_link,
            message=msg_text,
            email_sent=email_ok
        )

        return {
            "notification_id": notif_id,
            "dashboard_created": True,
            "email_sent": email_ok,
            "email_error": email_err
        }

