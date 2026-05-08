"""Dataset loading and random selection helpers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TestCase:
    id: str
    question: str
    type: str
    difficulty: str
    keywords: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DatasetSelection:
    cases: list[TestCase]
    fresh_selected_count: int
    repeated_questions_avoided: int
    reused_questions_count: int


def _load_raw_items(file: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    path = Path(file).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw_items = json.load(handle)

    if not isinstance(raw_items, list):
        raise ValueError(f"Dataset must be a JSON list: {path}")
    return path, raw_items


def _case_id(item: dict[str, Any]) -> str:
    return str(item["id"]).strip()


def _to_test_case(item: dict[str, Any]) -> TestCase:
    return TestCase(
        id=_case_id(item),
        question=str(item["question"]).strip(),
        type=str(item.get("type", item.get("category", "general"))).strip().lower(),
        difficulty=str(item.get("difficulty", "medium")).strip().lower(),
        keywords=[
            str(keyword).strip()
            for keyword in item.get("keywords", item.get("expected_keywords", []))
        ],
    )


def load_dataset(
    file: str | Path,
    limit: int | None = 25,
    excluded_case_ids: set[str] | None = None,
    hard_excluded_case_ids: set[str] | None = None,
) -> DatasetSelection:
    """Load a dataset and return a true-random sample with recent-repeat avoidance.

    Notes:
    - Questions are randomized on every run.
    - Recent case IDs in ``excluded_case_ids`` are skipped first.
    - If there are not enough fresh questions, older questions are brought
      back in random order to fill the requested limit.
    - If ``limit`` is ``None``, all available questions are returned in
      random order.
    """

    _, raw_items = _load_raw_items(file)
    excluded_ids = {str(case_id).strip() for case_id in (excluded_case_ids or set())}
    hard_excluded_ids = {str(case_id).strip() for case_id in (hard_excluded_case_ids or set())}
    fresh_items = [
        item for item in raw_items
        if _case_id(item) not in excluded_ids and _case_id(item) not in hard_excluded_ids
    ]
    stale_items = [
        item for item in raw_items
        if _case_id(item) in excluded_ids and _case_id(item) not in hard_excluded_ids
    ]

    sampler = random.SystemRandom()
    sampler.shuffle(fresh_items)
    sampler.shuffle(stale_items)

    if limit is None:
        selected_fresh = fresh_items
        selected_reused = stale_items
    else:
        selected_fresh = fresh_items[:limit]
        selected_reused = stale_items[: max(limit - len(selected_fresh), 0)]

    selected_items = [*selected_fresh, *selected_reused]
    avoided_ids = excluded_ids.union(hard_excluded_ids)
    repeated_questions_avoided = sum(1 for item in raw_items if _case_id(item) in avoided_ids)

    return DatasetSelection(
        cases=[_to_test_case(item) for item in selected_items],
        fresh_selected_count=len(selected_fresh),
        repeated_questions_avoided=repeated_questions_avoided,
        reused_questions_count=len(selected_reused),
    )


def recent_question_ids(
    history_file: str | Path,
    cooldown_minutes: int,
    now: datetime | None = None,
) -> set[str]:
    """Return case IDs seen within the cooldown window."""

    if cooldown_minutes <= 0:
        return set()

    path = Path(history_file).resolve()
    if not path.exists():
        return set()

    now_utc = now or datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(minutes=cooldown_minutes)

    try:
        raw_items = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    if not isinstance(raw_items, list):
        return set()

    recent_ids: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        timestamp_text = str(item.get("selected_at") or "").strip()
        case_id = item.get("case_id")
        if not timestamp_text or case_id is None:
            continue
        try:
            selected_at = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
        except ValueError:
            continue
        if selected_at.tzinfo is None:
            selected_at = selected_at.replace(tzinfo=timezone.utc)
        if selected_at >= cutoff:
            normalized_case_id = str(case_id).strip()
            if normalized_case_id:
                recent_ids.add(normalized_case_id)
    return recent_ids


def append_case_selection_history(
    history_file: str | Path,
    dataset_file: str | Path,
    cases: list[TestCase],
    selected_at: datetime | None = None,
    retention_days: int = 7,
) -> None:
    """Persist selected question IDs so near-future runs can avoid them."""

    path = Path(history_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    now_utc = selected_at or datetime.now(timezone.utc)
    retention_cutoff = now_utc - timedelta(days=max(retention_days, 1))

    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = [item for item in loaded if isinstance(item, dict)]
        except Exception:
            existing = []

    retained: list[dict[str, Any]] = []
    for item in existing:
        timestamp_text = str(item.get("selected_at") or "").strip()
        if not timestamp_text:
            continue
        try:
            timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
        except ValueError:
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if timestamp >= retention_cutoff:
            retained.append(item)

    dataset_name = Path(dataset_file).resolve().name
    selected_at_text = now_utc.isoformat()
    for case in cases:
        retained.append(
            {
                "selected_at": selected_at_text,
                "dataset": dataset_name,
                "case_id": case.id,
                "question": case.question,
            }
        )

    path.write_text(json.dumps(retained, indent=2, ensure_ascii=False), encoding="utf-8")


def count_fresh_cases(
    file: str | Path,
    excluded_case_ids: set[str] | None = None,
    hard_excluded_case_ids: set[str] | None = None,
) -> int:
    """Return how many cases are currently outside the recent-repeat window."""

    _, raw_items = _load_raw_items(file)
    excluded_ids = {str(case_id).strip() for case_id in (excluded_case_ids or set())}
    hard_excluded_ids = {str(case_id).strip() for case_id in (hard_excluded_case_ids or set())}
    return sum(
        1
        for item in raw_items
        if _case_id(item) not in excluded_ids and _case_id(item) not in hard_excluded_ids
    )
