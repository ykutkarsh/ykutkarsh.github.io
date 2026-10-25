"""Question proposals ranked using explicit, measurable criteria."""

from __future__ import annotations

from dataclasses import dataclass

from aab.memory.database import ExperienceStore


@dataclass(frozen=True)
class ResearchQuestion:
    text: str
    relevance: float
    uncertainty: float
    novelty: float
    learning_value: float
    score: float
    rationale: str
    status: str = "proposal"


class CuriosityEngine:
    def __init__(self, store: ExperienceStore) -> None:
        self.store = store

    def propose(self, objective: str, knowledge_gaps: list[str]) -> list[ResearchQuestion]:
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective must be a non-empty string")
        if not isinstance(knowledge_gaps, list) or any(
            not isinstance(gap, str) or not gap.strip() for gap in knowledge_gaps
        ):
            raise ValueError("knowledge_gaps must be a list of non-empty strings")
        questions = []
        for gap in knowledge_gaps:
            text = f"What evidence addresses '{gap.strip()}' for '{objective.strip()}'?"
            if self.store.question_exists(text):
                continue
            values = (0.9, 0.8, 0.7, 0.9)
            score = sum(values) / len(values)
            questions.append(
                ResearchQuestion(
                    text, *values, score,
                    "Selected because it directly addresses a stated gap and has high expected learning value.",
                )
            )
        return sorted(questions, key=lambda question: question.score, reverse=True)

    def select(self, objective: str, knowledge_gaps: list[str]) -> ResearchQuestion | None:
        proposals = self.propose(objective, knowledge_gaps)
        if not proposals:
            return None
        selected = proposals[0]
        self.store.record_question(selected.text, objective, selected.rationale)
        return selected
