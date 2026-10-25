"""SQLite persistence for structured learning experiences."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class ExperienceStore:
    """Store predictions, outcomes, provenance, and evaluation metrics."""

    def __init__(self, path: str | Path = "data/experiences.db") -> None:
        self.path = Path(path)
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                strategy TEXT NOT NULL,
                prediction TEXT NOT NULL,
                outcome TEXT NOT NULL,
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                source TEXT,
                verification_status TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.commit()

    def record(
        self,
        *,
        task: str,
        strategy: str,
        prediction: str,
        outcome: str,
        confidence: float,
        source: str | None,
        verification_status: str,
        metrics_json: str,
    ) -> int:
        """Record one measurable experience and return its database ID."""
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        cursor = self._connection.execute(
            """
            INSERT INTO experiences
                (task, strategy, prediction, outcome, confidence, source,
                 verification_status, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task,
                strategy,
                prediction,
                outcome,
                confidence,
                source,
                verification_status,
                metrics_json,
            ),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def get(self, experience_id: int) -> dict[str, Any] | None:
        """Return an experience as a dictionary, if it exists."""
        row = self._connection.execute(
            "SELECT * FROM experiences WHERE id = ?", (experience_id,)
        ).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        """Return the number of stored experiences."""
        return int(
            self._connection.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ExperienceStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
