"""
Contract Lifecycle Management Subsystem (Agent + Tools).

Manages the full contract lifecycle:
- Contract creation and versioning
- Deadline and renewal tracking
- Obligation monitoring
- Amendment workflow
- Expiry alerts
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import sqlite3
import json
import hashlib
import re
import logging
from pathlib import Path

logger = logging.getLogger("lifecycle_mgmt")


class LifecycleManagementSystem:
    """Agent-based contract lifecycle management.

    Uses tools for:
    - Database operations (store, retrieve, update contracts)
    - Calendar integration (deadlines, renewals)
    - Email notifications (alerts, reminders)
    - Web search (regulation updates)

    Design:
    - SQLite-backed for zero infrastructure costs
    - Event-driven alerts for expiring contracts
    - Version tracking for amendments
    - Obligation monitoring with status tracking
    """

    def __init__(self, llm=None, kb=None):
        self.llm = llm
        self.kb = kb
        self.conn = sqlite3.connect("data/audit.db", check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        """Create lifecycle management tables."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS contracts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                language TEXT DEFAULT 'english',
                parties TEXT DEFAULT '[]',
                effective_date TEXT,
                expiry_date TEXT,
                status TEXT DEFAULT 'active',
                value REAL DEFAULT 0,
                currency TEXT DEFAULT 'AED',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS obligations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT NOT NULL,
                obligation TEXT NOT NULL,
                party TEXT NOT NULL,
                due_date TEXT,
                status TEXT DEFAULT 'pending',
                priority TEXT DEFAULT 'medium',
                notes TEXT DEFAULT '',
                FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
            );

            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                content TEXT NOT NULL,
                change_summary TEXT DEFAULT '',
                created_by TEXT DEFAULT 'system',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                due_date TEXT,
                acknowledged INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
            );
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Contract Management Tools
    # ------------------------------------------------------------------

    def register_contract(self, title: str, contract_type: str,
                          parties: List[str], effective_date: str = "",
                          expiry_date: str = "", value: float = 0,
                          language: str = "english") -> Dict:
        """Register a new contract in the system."""
        contract_id = f"CT-{datetime.now().strftime('%Y%m%d')}-{hashlib.md5(title.encode()).hexdigest()[:4].upper()}"

        try:
            self.conn.execute("""
                INSERT INTO contracts
                (contract_id, title, type, language, parties,
                 effective_date, expiry_date, value, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
            """, (contract_id, title, contract_type, language,
                  json.dumps(parties), effective_date, expiry_date, value))
            self.conn.commit()

            # Create initial version entry
            self.conn.execute("""
                INSERT INTO versions (contract_id, version, content, change_summary)
                VALUES (?, 1, 'Contract registered', 'Initial registration')
            """, (contract_id,))
            self.conn.commit()

            # Create alerts for expiring contracts
            if expiry_date:
                self._create_expiry_alert(contract_id, title, expiry_date)

            logger.info(f"Contract registered: {contract_id} - {title}")
            return {
                "contract_id": contract_id,
                "title": title,
                "status": "active",
                "created_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Contract registration error: {e}")
            return {"error": str(e)}

    def get_contract(self, contract_id: str) -> Optional[Dict]:
        """Retrieve contract details."""
        cursor = self.conn.execute(
            "SELECT * FROM contracts WHERE contract_id = ?", (contract_id,)
        )
        row = cursor.fetchone()
        if row:
            cols = [d[0] for d in cursor.description]
            return dict(zip(cols, row))
        return None

    def list_contracts(self, status: str = "", contract_type: str = "") -> List[Dict]:
        """List contracts with optional filters."""
        query = "SELECT * FROM contracts WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if contract_type:
            query += " AND type = ?"
            params.append(contract_type)

        query += " ORDER BY created_at DESC LIMIT 50"

        cursor = self.conn.execute(query, params)
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def update_status(self, contract_id: str, status: str) -> bool:
        """Update contract status (active, expired, terminated, amended)."""
        try:
            self.conn.execute("""
                UPDATE contracts SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE contract_id = ?
            """, (status, contract_id))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Status update error: {e}")
            return False

    # ------------------------------------------------------------------
    # Obligation Tracking
    # ------------------------------------------------------------------

    def add_obligation(self, contract_id: str, obligation: str,
                       party: str, due_date: str = "",
                       priority: str = "medium") -> Dict:
        """Add an obligation to track."""
        try:
            self.conn.execute("""
                INSERT INTO obligations
                (contract_id, obligation, party, due_date, priority, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
            """, (contract_id, obligation, party, due_date, priority))
            self.conn.commit()
            return {"success": True, "obligation": obligation, "party": party}
        except Exception as e:
            logger.error(f"Obligation add error: {e}")
            return {"error": str(e)}

    def get_obligations(self, contract_id: str, status: str = "") -> List[Dict]:
        """Get obligations for a contract."""
        query = "SELECT * FROM obligations WHERE contract_id = ?"
        params = [contract_id]

        if status:
            query += " AND status = ?"
            params.append(status)

        cursor = self.conn.execute(query, params)
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def update_obligation_status(self, obligation_id: int, status: str,
                                  notes: str = "") -> bool:
        """Mark an obligation as fulfilled, waived, or overdue."""
        try:
            self.conn.execute("""
                UPDATE obligations SET status = ?, notes = ?
                WHERE id = ?
            """, (status, notes, obligation_id))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Obligation update error: {e}")
            return False

    # ------------------------------------------------------------------
    # Version Control
    # ------------------------------------------------------------------

    def create_version(self, contract_id: str, content: str,
                       change_summary: str = "", created_by: str = "system") -> Dict:
        """Create a new version of the contract."""
        try:
            # Get current version number
            cursor = self.conn.execute(
                "SELECT MAX(version) FROM versions WHERE contract_id = ?",
                (contract_id,)
            )
            current_version = cursor.fetchone()[0] or 0
            new_version = current_version + 1

            self.conn.execute("""
                INSERT INTO versions
                (contract_id, version, content, change_summary, created_by)
                VALUES (?, ?, ?, ?, ?)
            """, (contract_id, new_version, content, change_summary, created_by))
            self.conn.commit()

            return {"version": new_version, "change_summary": change_summary}
        except Exception as e:
            logger.error(f"Version creation error: {e}")
            return {"error": str(e)}

    def get_versions(self, contract_id: str) -> List[Dict]:
        """Get version history for a contract."""
        cursor = self.conn.execute(
            "SELECT * FROM versions WHERE contract_id = ? ORDER BY version DESC",
            (contract_id,)
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Alerts & Notifications
    # ------------------------------------------------------------------

    def _create_expiry_alert(self, contract_id: str, title: str,
                              expiry_date: str):
        """Create alerts for approaching expiry."""
        try:
            expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
            today = datetime.now()

            # Alert 30 days before
            alert_30 = expiry - timedelta(days=30)
            if alert_30 > today:
                self.conn.execute("""
                    INSERT INTO alerts (contract_id, alert_type, message, due_date)
                    VALUES (?, 'renewal_reminder', ?, ?)
                """, (contract_id,
                      f"Contract '{title}' expires in 30 days on {expiry_date}",
                      alert_30.strftime("%Y-%m-%d")))

            # Alert 7 days before
            alert_7 = expiry - timedelta(days=7)
            if alert_7 > today:
                self.conn.execute("""
                    INSERT INTO alerts (contract_id, alert_type, message, due_date)
                    VALUES (?, 'expiry_warning', ?, ?)
                """, (contract_id,
                      f"URGENT: Contract '{title}' expires in 7 days on {expiry_date}",
                      alert_7.strftime("%Y-%m-%d")))

            self.conn.commit()
        except Exception as e:
            logger.warning(f"Alert creation error: {e}")

    def get_alerts(self, contract_id: str = "", unacknowledged_only: bool = True) -> List[Dict]:
        """Get alerts, optionally filtered."""
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []

        if contract_id:
            query += " AND contract_id = ?"
            params.append(contract_id)
        if unacknowledged_only:
            query += " AND acknowledged = 0"

        query += " ORDER BY due_date ASC"

        cursor = self.conn.execute(query, params)
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Acknowledge an alert."""
        try:
            self.conn.execute(
                "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
                (alert_id,)
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Alert acknowledge error: {e}")
            return False

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_dashboard_stats(self) -> Dict:
        """Get lifecycle dashboard statistics."""
        stats = {}

        # Contract counts by status
        cursor = self.conn.execute(
            "SELECT status, COUNT(*) FROM contracts GROUP BY status"
        )
        stats["contracts_by_status"] = dict(cursor.fetchall())

        # Total contracts
        stats["total_contracts"] = sum(stats["contracts_by_status"].values()) if stats.get("contracts_by_status") else 0

        # Pending obligations
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM obligations WHERE status = 'pending'"
        )
        stats["pending_obligations"] = cursor.fetchone()[0]

        # Unacknowledged alerts
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE acknowledged = 0"
        )
        stats["unacknowledged_alerts"] = cursor.fetchone()[0]

        # Expiring soon (next 30 days)
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM contracts WHERE expiry_date BETWEEN date('now') AND date('now', '+30 days')"
        )
        stats["expiring_soon"] = cursor.fetchone()[0]

        return stats
