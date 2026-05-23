"""
Database tool for contract storage, querying, and management.

Provides:
- Contract CRUD operations
- Full-text search on contracts
- Metadata queries by type, status, date
- Version history access
"""

from typing import Dict, List, Optional, Any
import sqlite3
import json
import logging
from datetime import datetime

logger = logging.getLogger("tool_database")


class DatabaseTool:
    """SQLite-based database tool for contract management.

    Design:
    - Zero infrastructure costs (file-based SQLite)
    - Extensible to PostgreSQL for production
    - Full-text search for contract content
    - Optimized queries with proper indexing
    """

    def __init__(self, db_path: str = "data/audit.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Ensure required tables exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS contracts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                language TEXT DEFAULT 'english',
                content TEXT DEFAULT '',
                parties TEXT DEFAULT '[]',
                effective_date TEXT,
                expiry_date TEXT,
                status TEXT DEFAULT 'active',
                value REAL DEFAULT 0,
                currency TEXT DEFAULT 'AED',
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_contracts_status ON contracts(status);
            CREATE INDEX IF NOT EXISTS idx_contracts_type ON contracts(type);
            CREATE INDEX IF NOT EXISTS idx_contracts_expiry ON contracts(expiry_date);
        """)
        self.conn.commit()

    def insert_contract(self, contract_id: str, title: str, type_: str,
                       content: str = "", parties: List[str] = None,
                       effective_date: str = "", expiry_date: str = "",
                       value: float = 0, language: str = "english",
                       metadata: Dict = None) -> bool:
        """Insert a new contract record."""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO contracts
                (contract_id, title, type, language, content, parties,
                 effective_date, expiry_date, value, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (contract_id, title, type_, language, content,
                  json.dumps(parties or []), effective_date, expiry_date,
                  value, json.dumps(metadata or {})))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"DB insert error: {e}")
            return False

    def get_contract(self, contract_id: str) -> Optional[Dict]:
        """Retrieve a contract by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM contracts WHERE contract_id = ?", (contract_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def search_contracts(self, query: str, field: str = "all",
                        limit: int = 20) -> List[Dict]:
        """Search contracts by content, title, or parties."""
        if field == "all":
            cursor = self.conn.execute("""
                SELECT * FROM contracts WHERE
                title LIKE ? OR content LIKE ? OR parties LIKE ?
                ORDER BY created_at DESC LIMIT ?
            """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
        elif field == "title":
            cursor = self.conn.execute(
                "SELECT * FROM contracts WHERE title LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", limit))
        elif field == "content":
            cursor = self.conn.execute(
                "SELECT * FROM contracts WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", limit))
        else:
            return []

        return [dict(row) for row in cursor.fetchall()]

    def list_contracts(self, status: str = "", type_: str = "",
                      limit: int = 50) -> List[Dict]:
        """List contracts with optional filters."""
        sql = "SELECT * FROM contracts WHERE 1=1"
        params = []

        if status:
            sql += " AND status = ?"
            params.append(status)
        if type_:
            sql += " AND type = ?"
            params.append(type_)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def update_status(self, contract_id: str, status: str) -> bool:
        """Update contract status."""
        try:
            self.conn.execute(
                "UPDATE contracts SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE contract_id = ?",
                (status, contract_id))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Status update error: {e}")
            return False

    def get_expiring_contracts(self, days: int = 30) -> List[Dict]:
        """Get contracts expiring within specified days."""
        cursor = self.conn.execute("""
            SELECT * FROM contracts WHERE
            expiry_date BETWEEN date('now') AND date('now', ?)
            AND status = 'active'
            ORDER BY expiry_date ASC
        """, (f"+{days} days",))
        return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict:
        """Get database statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        by_status = {}
        cursor = self.conn.execute("SELECT status, COUNT(*) FROM contracts GROUP BY status")
        for row in cursor:
            by_status[row[0]] = row[1]
        return {"total_contracts": total, "by_status": by_status}
