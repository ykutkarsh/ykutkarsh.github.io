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
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL UNIQUE,
                objective TEXT NOT NULL,
                rationale TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'proposed',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                objective TEXT NOT NULL,
                selected_action TEXT NOT NULL,
                rationale TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS autonomy_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_key TEXT NOT NULL UNIQUE,
                objective TEXT NOT NULL,
                question TEXT NOT NULL,
                rationale TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                outcome TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS discoveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                conflict_group TEXT,
                citations_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                discovery_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                quality REAL NOT NULL CHECK (quality >= 0 AND quality <= 1),
                reasons_json TEXT NOT NULL,
                FOREIGN KEY (discovery_id) REFERENCES discoveries(id)
            );
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

    def record_question(self, question: str, objective: str, rationale: str) -> int:
        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO research_questions (question, objective, rationale) VALUES (?, ?, ?)",
            (question, objective, rationale),
        )
        self._connection.commit()
        return int(cursor.lastrowid or self._connection.execute(
            "SELECT id FROM research_questions WHERE question = ?", (question,)
        ).fetchone()[0])

    def question_completed(self, question: str) -> bool:
        row = self._connection.execute(
            "SELECT status FROM research_questions WHERE question = ?", (question,)
        ).fetchone()
        return row is not None and row["status"] == "completed"

    def question_exists(self, question: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM research_questions WHERE question = ?", (question,)
        ).fetchone()
        return row is not None

    def record_decision(self, objective: str, selected_action: str, rationale: str) -> int:
        cursor = self._connection.execute(
            "INSERT INTO decisions (objective, selected_action, rationale) VALUES (?, ?, ?)",
            (objective, selected_action, rationale),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def create_task(
        self, task_key: str, objective: str, question: str, rationale: str
    ) -> int | None:
        cursor = self._connection.execute(
            """
            INSERT OR IGNORE INTO autonomy_tasks
                (task_key, objective, question, rationale)
            VALUES (?, ?, ?, ?)
            """,
            (task_key, objective, question, rationale),
        )
        self._connection.commit()
        if cursor.rowcount == 0:
            return None
        return int(cursor.lastrowid)

    def update_task_status(self, task_id: int, status: str, outcome: str | None = None) -> None:
        allowed = {"pending", "running", "completed", "failed"}
        if status not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        self._connection.execute(
            """
            UPDATE autonomy_tasks
            SET status = ?, outcome = COALESCE(?, outcome), updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status, outcome, task_id),
        )
        self._connection.commit()

    def task_count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM autonomy_tasks").fetchone()[0])

    def record_discovery(
        self,
        claim: str,
        status: str,
        confidence: float,
        conflict_group: str | None,
        citations: tuple[str, ...],
    ) -> int:
        import json

        cursor = self._connection.execute(
            """
            INSERT INTO discoveries
                (claim, status, confidence, conflict_group, citations_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (claim, status, confidence, conflict_group, json.dumps(citations)),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def record_citations(self, discovery_id: int, documents: list[Any], assessments: list[Any]) -> None:
        import json

        self._connection.executemany(
            """
            INSERT INTO citations
                (discovery_id, url, title, retrieved_at, quality, reasons_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    discovery_id,
                    document.url,
                    document.title,
                    document.retrieved_at,
                    assessment.quality,
                    json.dumps(assessment.reasons),
                )
                for document, assessment in zip(documents, assessments)
            ],
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ExperienceStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
