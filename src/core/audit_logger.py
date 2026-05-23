"""
HIPAA-compliant audit logging with trace_id for full request tracing.
"""

import sqlite3
import json
import hashlib
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger("legal_audit")


class AuditLogger:
    """Comprehensive audit logging with trace IDs.

    Every action gets:
    - Unique trace_id linking user action to AI response to human review
    - Timestamp, duration, model used, tokens consumed
    - Success/failure status with error messages
    - Support for human review records (lawyer override)
    """

    def __init__(self, db_path: str = "data/audit.db"):
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        """Create audit tables with proper schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                subsystem TEXT DEFAULT '',
                user TEXT DEFAULT 'anonymous',
                role TEXT DEFAULT 'user',
                contract_type TEXT DEFAULT '',
                language TEXT DEFAULT '',
                success BOOLEAN DEFAULT 1,
                duration_ms REAL DEFAULT 0,
                provider TEXT DEFAULT 'google',
                model_used TEXT DEFAULT '',
                tokens_estimated INTEGER DEFAULT 0,
                summary TEXT DEFAULT '',
                error TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS human_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                review_timestamp TEXT NOT NULL,
                decision TEXT NOT NULL,
                notes TEXT DEFAULT '',
                changes_required TEXT DEFAULT '',
                FOREIGN KEY (trace_id) REFERENCES audit_trail(trace_id)
            );

            CREATE INDEX IF NOT EXISTS idx_audit_trace ON audit_trail(trace_id);
            CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_trail(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_trail(user);
        """)
        self.conn.commit()

    def log(self, trace_id: str, action: str, subsystem: str = "",
            user: str = "anonymous", role: str = "user",
            contract_type: str = "", language: str = "",
            success: bool = True, duration_ms: float = 0,
            provider: str = "google", model_used: str = "",
            tokens_estimated: int = 0, summary: str = "",
            error: str = "") -> str:
        """Log an action with full trace context."""
        try:
            self.conn.execute("""
                INSERT OR IGNORE INTO audit_trail
                (trace_id, timestamp, action, subsystem, user, role,
                 contract_type, language, success, duration_ms,
                 provider, model_used, tokens_estimated, summary, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace_id, datetime.now().isoformat(), action, subsystem,
                user, role, contract_type, language, success,
                round(duration_ms, 1), provider, model_used,
                tokens_estimated, str(summary)[:500], str(error)[:500]
            ))
            self.conn.commit()
            return trace_id
        except Exception as e:
            logger.error(f"Audit log failed: {e}")
            return trace_id

    def log_review(self, trace_id: str, reviewer: str, decision: str,
                   notes: str = "", changes: str = "") -> bool:
        """Log a human review of an AI action."""
        try:
            self.conn.execute("""
                INSERT INTO human_reviews
                (trace_id, reviewer, review_timestamp, decision, notes, changes_required)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (trace_id, reviewer, datetime.now().isoformat(),
                  decision, notes, changes))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Review log failed: {e}")
            return False

    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """Get full trace details."""
        cursor = self.conn.execute(
            "SELECT * FROM audit_trail WHERE trace_id = ?", (trace_id,)
        )
        row = cursor.fetchone()
        if row:
            columns = [d[0] for d in cursor.description]
            return dict(zip(columns, row))
        return None

    def get_reviews(self, trace_id: str) -> list:
        """Get all human reviews for a trace."""
        cursor = self.conn.execute(
            "SELECT * FROM human_reviews WHERE trace_id = ? ORDER BY review_timestamp",
            (trace_id,)
        )
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_stats(self) -> Dict:
        """Get audit statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM audit_trail").fetchone()[0]
        successful = self.conn.execute(
            "SELECT COUNT(*) FROM audit_trail WHERE success = 1"
        ).fetchone()[0]
        reviews = self.conn.execute(
            "SELECT COUNT(*) FROM human_reviews"
        ).fetchone()[0]

        return {
            "total_actions": total,
            "successful": successful,
            "success_rate": round(successful / total * 100, 1) if total > 0 else 100,
            "human_reviews": reviews,
        }

    def get_recent(self, limit: int = 20) -> list:
        """Get recent audit entries."""
        cursor = self.conn.execute(
            "SELECT * FROM audit_trail ORDER BY id DESC LIMIT ?", (limit,)
        )
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
