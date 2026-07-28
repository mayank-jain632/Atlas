from __future__ import annotations

import json
from pathlib import Path

import pytest

from live.execution.run_log import RunLog


@pytest.fixture
def run_log(tmp_path: Path):
    with RunLog(db_path=tmp_path / "run_log.sqlite") as log:
        yield log


def test_record_and_get_latest(run_log: RunLog) -> None:
    run_log.record("uid-1", as_of="2026-01-05", decision="REBALANCED")

    latest = run_log.get_latest("uid-1")
    assert latest is not None
    assert latest["uid"] == "uid-1"
    assert latest["decision"] == "REBALANCED"


def test_get_latest_for_unknown_uid_is_none(run_log: RunLog) -> None:
    assert run_log.get_latest("never-ran") is None


def test_records_every_check_even_with_no_action(run_log: RunLog) -> None:
    run_log.record("uid-1", as_of="2026-01-05", decision="REBALANCED")
    run_log.record("uid-1", as_of="2026-01-06", decision="NO_REBALANCE")
    run_log.record("uid-1", as_of="2026-01-07", decision="NO_REBALANCE")

    history = run_log.get_history("uid-1")
    assert len(history) == 3
    # most recent first
    assert list(history["decision"]) == ["NO_REBALANCE", "NO_REBALANCE", "REBALANCED"]


def test_get_history_respects_limit(run_log: RunLog) -> None:
    for day in range(1, 6):
        run_log.record("uid-1", as_of=f"2026-01-{day:02d}", decision="NO_REBALANCE")

    assert len(run_log.get_history("uid-1")) == 5
    assert len(run_log.get_history("uid-1", limit=2)) == 2


def test_different_uids_do_not_collide(run_log: RunLog) -> None:
    run_log.record("uid-a", as_of="2026-01-05", decision="REBALANCED")
    run_log.record("uid-b", as_of="2026-01-05", decision="ERROR")

    assert run_log.get_latest("uid-a")["decision"] == "REBALANCED"
    assert run_log.get_latest("uid-b")["decision"] == "ERROR"


def test_record_carries_data_as_of_and_detail(run_log: RunLog) -> None:
    run_log.record(
        "uid-1",
        as_of="2026-01-05",
        decision="ERROR",
        data_as_of="2026-01-05T16:00:00",
        detail={"error": "connection timed out"},
    )

    latest = run_log.get_latest("uid-1")
    assert latest["data_as_of"] is not None
    assert "2026-01-05" in latest["data_as_of"]
    assert json.loads(latest["detail"]) == {"error": "connection timed out"}


def test_record_without_data_as_of_or_detail_defaults_cleanly(
    run_log: RunLog,
) -> None:
    run_log.record("uid-1", as_of="2026-01-05", decision="NO_REBALANCE")

    latest = run_log.get_latest("uid-1")
    assert latest["data_as_of"] is None
    assert json.loads(latest["detail"]) == {}


def test_get_all_history_spans_every_uid_most_recent_first(run_log: RunLog) -> None:
    run_log.record("uid-a", as_of="2026-01-05", decision="REBALANCED")
    run_log.record("uid-b", as_of="2026-01-05", decision="ERROR")
    run_log.record("uid-a", as_of="2026-01-06", decision="NO_REBALANCE")

    history = run_log.get_all_history()
    assert len(history) == 3
    assert list(history["uid"]) == ["uid-a", "uid-b", "uid-a"]


def test_get_all_history_filters_by_decision(run_log: RunLog) -> None:
    run_log.record("uid-a", as_of="2026-01-05", decision="REBALANCED")
    run_log.record("uid-b", as_of="2026-01-05", decision="ERROR")
    run_log.record("uid-c", as_of="2026-01-05", decision="ERROR")

    errors = run_log.get_all_history(decision="ERROR")
    assert len(errors) == 2
    assert set(errors["uid"]) == {"uid-b", "uid-c"}


def test_get_all_history_respects_limit(run_log: RunLog) -> None:
    for day in range(1, 6):
        run_log.record("uid-a", as_of=f"2026-01-{day:02d}", decision="NO_REBALANCE")

    assert len(run_log.get_all_history()) == 5
    assert len(run_log.get_all_history(limit=2)) == 2


def test_get_latest_per_uid_returns_one_row_per_uid(run_log: RunLog) -> None:
    run_log.record("uid-a", as_of="2026-01-05", decision="REBALANCED")
    run_log.record("uid-a", as_of="2026-01-06", decision="NO_REBALANCE")
    run_log.record("uid-b", as_of="2026-01-05", decision="ERROR")

    latest = run_log.get_latest_per_uid().set_index("uid")
    assert len(latest) == 2
    assert latest.loc["uid-a", "decision"] == "NO_REBALANCE"
    assert latest.loc["uid-b", "decision"] == "ERROR"


def test_get_latest_per_uid_empty_when_nothing_recorded(run_log: RunLog) -> None:
    assert run_log.get_latest_per_uid().empty


def test_list_uids_is_alphabetical_and_deduplicated(run_log: RunLog) -> None:
    run_log.record("uid-b", as_of="2026-01-05", decision="REBALANCED")
    run_log.record("uid-a", as_of="2026-01-05", decision="NO_REBALANCE")
    run_log.record("uid-a", as_of="2026-01-06", decision="NO_REBALANCE")

    assert run_log.list_uids() == ["uid-a", "uid-b"]


def test_list_uids_is_empty_for_a_fresh_log(run_log: RunLog) -> None:
    assert run_log.list_uids() == []


def test_persists_across_reopening_the_same_db_file(tmp_path: Path) -> None:
    db_path = tmp_path / "run_log.sqlite"

    with RunLog(db_path=db_path) as log:
        log.record("uid-1", as_of="2026-01-05", decision="REBALANCED")

    with RunLog(db_path=db_path) as reopened:
        assert reopened.get_latest("uid-1")["decision"] == "REBALANCED"


def test_default_db_path_matches_signal_books_default() -> None:
    # RunLog imports SignalBook's DEFAULT_SIGNAL_BOOK_PATH directly rather
    # than defining its own -- that's what guarantees the two classes
    # always agree on where they live, not just happen to match today.
    from live.execution.run_log import DEFAULT_SIGNAL_BOOK_PATH as run_log_default
    from live.execution.signal_book import (
        DEFAULT_SIGNAL_BOOK_PATH as signal_book_default,
    )

    assert run_log_default == signal_book_default
