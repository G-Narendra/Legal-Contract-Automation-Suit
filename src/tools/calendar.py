"""
Calendar integration tool for contract deadlines and renewals.

Manages:
- Contract effective/expiry dates
- Renewal reminders
- Obligation due dates
- Review schedules
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("tool_calendar")


class CalendarTool:
    """Calendar integration for legal contract lifecycle events.

    Design:
    - Local SQLite-backed storage (no external API costs)
    - Extensible to Google Calendar / Outlook integration
    - Event-based reminders with configurable lead times
    """

    def __init__(self, db_connection=None):
        self.db = db_connection
        if self.db:
            self._create_tables()

    def _create_tables(self):
        """Ensure calendar tables exist."""
        try:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS calendar_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    event_date TEXT NOT NULL,
                    event_type TEXT DEFAULT 'deadline',
                    description TEXT DEFAULT '',
                    contract_id TEXT DEFAULT '',
                    reminder_days INTEGER DEFAULT 7,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.db.commit()
        except Exception as e:
            logger.warning(f"Calendar table creation error: {e}")

    def create_event(self, title: str, event_date: str,
                     event_type: str = "deadline",
                     description: str = "",
                     reminder_days: int = 7) -> Dict:
        """Create a calendar event.

        Args:
            title: Event title
            event_date: Date string (YYYY-MM-DD)
            event_type: 'deadline', 'renewal', 'review', 'reminder'
            description: Event description
            reminder_days: Days before to create reminder

        Returns:
            Dict with event details
        """
        event = {
            "title": title,
            "date": event_date,
            "type": event_type,
            "description": description,
            "reminder_days": reminder_days,
            "created_at": datetime.now().isoformat(),
        }

        if self.db:
            try:
                self.db.execute(
                    """INSERT INTO calendar_events (title, event_date, event_type, description)
                       VALUES (?, ?, ?, ?)""",
                    (title, event_date, event_type, description),
                )
                self.db.commit()
            except Exception as e:
                logger.warning(f"Calendar DB error: {e}")

        logger.info(f"Calendar event created: {title} on {event_date}")
        return event

    def get_upcoming_events(self, days: int = 30) -> List[Dict]:
        """Get upcoming events within the specified days."""
        if not self.db:
            return []

        try:
            cursor = self.db.execute(
                """SELECT * FROM calendar_events
                   WHERE event_date BETWEEN date('now') AND date('now', ?)
                   ORDER BY event_date ASC""",
                (f"+{days} days",),
            )
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Calendar query error: {e}")
            return []

    def get_events_by_contract(self, contract_id: str) -> List[Dict]:
        """Get all calendar events for a specific contract."""
        if not self.db:
            return []

        try:
            cursor = self.db.execute(
                "SELECT * FROM calendar_events WHERE contract_id = ? ORDER BY event_date",
                (contract_id,),
            )
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            return []
