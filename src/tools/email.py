"""
Email notification tool for contract alerts and reminders.

Sends:
- Renewal reminders to parties
- Expiry warnings to legal team
- Obligation due notifications
- Review completion notices
"""

from typing import Dict, List, Optional
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger("tool_email")


class EmailTool:
    """Email notification tool for contract lifecycle events.

    Design:
    - SMTP-based (works with any email provider)
    - Templates for common notification types
    - Batch sending for efficiency
    - Fallback to logging when SMTP not configured
    """

    def __init__(self, smtp_server: str = "", smtp_port: int = 587,
                 username: str = "", password: str = "",
                 from_address: str = "noreply@legalautomation.ae"):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_address = from_address
        self._enabled = all([smtp_server, username, password])

    def send_notification(self, to: str, subject: str, body: str,
                          notification_type: str = "general") -> Dict:
        """Send an email notification.

        Returns:
            Dict with success status and message
        """
        if not self._enabled:
            logger.info(f"[EMAIL DISABLED] Would send to {to}: {subject}")
            return {"success": False, "message": "SMTP not configured", "logged": True}

        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_address
            msg["To"] = to
            msg["Subject"] = f"[Legal Automation] {subject}"

            # HTML body with styling
            html_body = f"""
            <html><body style="font-family: Arial, sans-serif; padding: 20px;">
                <div style="max-width: 600px; margin: 0 auto; border: 1px solid #ddd; border-radius: 8px;">
                    <div style="background: #1a237e; color: white; padding: 15px; border-radius: 8px 8px 0 0;">
                        <h2 style="margin: 0;">⚖️ Legal Automation Suite</h2>
                    </div>
                    <div style="padding: 20px;">
                        {body.replace(chr(10), '<br>')}
                        <hr style="margin: 20px 0;">
                        <p style="color: #666; font-size: 12px;">
                            This is an automated notification from Legal Contract Automation Suite.<br>
                            Sent: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                        </p>
                    </div>
                </div>
            </body></html>
            """
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email sent to {to}: {subject}")
            return {"success": True, "to": to, "subject": subject}

        except Exception as e:
            logger.error(f"Email send error: {e}")
            return {"success": False, "error": str(e)}

    def send_renewal_reminder(self, to: str, contract_title: str,
                              expiry_date: str, party_name: str = "") -> Dict:
        """Send a contract renewal reminder."""
        subject = f"Contract Renewal Reminder: {contract_title}"
        body = (
            f"<p>Dear {party_name or 'Valued Partner'},</p>"
            f"<p>This is a reminder that the contract <strong>{contract_title}</strong> "
            f"is expiring on <strong>{expiry_date}</strong>.</p>"
            f"<p>Please review and take appropriate action regarding renewal.</p>"
            f"<p><strong>Recommended actions:</strong><br>"
            f"1. Review contract terms<br>"
            f"2. Discuss renewal terms with counterparty<br>"
            f"3. Execute renewal or termination</p>"
        )
        return self.send_notification(to, subject, body, "renewal")

    def send_obligation_reminder(self, to: str, obligation: str,
                                 due_date: str, contract_title: str) -> Dict:
        """Send an obligation due reminder."""
        subject = f"Obligation Due: {obligation[:50]}"
        body = (
            f"<p>This is a reminder about the following obligation:</p>"
            f"<p><strong>Obligation:</strong> {obligation}<br>"
            f"<strong>Contract:</strong> {contract_title}<br>"
            f"<strong>Due Date:</strong> {due_date}</p>"
            f"<p>Please ensure this obligation is fulfilled by the due date.</p>"
        )
        return self.send_notification(to, subject, body, "obligation")

    def send_review_complete(self, to: str, contract_title: str,
                             risk_level: str, trace_id: str) -> Dict:
        """Send review completion notification."""
        subject = f"Review Complete: {contract_title}"
        body = (
            f"<p>The AI-assisted review of <strong>{contract_title}</strong> is complete.</p>"
            f"<p><strong>Risk Level:</strong> {risk_level.upper()}<br>"
            f"<strong>Trace ID:</strong> {trace_id}</p>"
            f"<p>Please log in to review the detailed analysis and provide your sign-off.</p>"
        )
        return self.send_notification(to, subject, body, "review")
