from __future__ import annotations

import json
from pathlib import Path

import pytest

from live.execution.signal_book import SignalBook


@pytest.fixture
def signal_book(tmp_path: Path):
    with SignalBook(db_path=tmp_path / "signal_book.sqlite") as book:
        yield book


def test_write_and_read_latest_targets(signal_book: SignalBook) -> None:
    signal_book.write_targets(
        uid="uid-1",
        targets={"AAPL": 10.0, "MSFT": 5.0},
        as_of="2026-01-05",
        reason="MOMENTUM_REBALANCE",
    )

    assert signal_book.get_latest_targets("uid-1") == {"AAPL": 10.0, "MSFT": 5.0}


def test_get_latest_targets_detail_includes_notes(signal_book: SignalBook) -> None:
    signal_book.write_targets(
        uid="uid-1",
        targets={"AAPL": 10.0, "MSFT": 5.0},
        as_of="2026-01-05",
        reason="MOMENTUM_REBALANCE",
        notes_by_symbol={
            "AAPL": {"target_weight": 0.12, "score": 0.5, "rank": 1},
            "MSFT": {"target_weight": 0.08, "score": 0.3, "rank": 2},
        },
    )

    detail = signal_book.get_latest_targets_detail("uid-1").set_index("symbol")

    assert list(detail.index) == ["AAPL", "MSFT"]
    assert detail.loc["AAPL", "target_quantity"] == 10.0

    aapl_notes = json.loads(detail.loc["AAPL", "notes"])
    assert aapl_notes["target_weight"] == 0.12
    assert aapl_notes["rank"] == 1


def test_get_latest_targets_detail_reflects_the_latest_write_only(
    signal_book: SignalBook,
) -> None:
    signal_book.write_targets(
        "uid-1", {"AAPL": 10.0}, as_of="2026-01-05", reason="R1",
        notes_by_symbol={"AAPL": {"target_weight": 0.10}},
    )
    signal_book.write_targets(
        "uid-1", {"AAPL": 12.0}, as_of="2026-02-05", reason="R2",
        notes_by_symbol={"AAPL": {"target_weight": 0.15}},
    )

    detail = signal_book.get_latest_targets_detail("uid-1").set_index("symbol")
    assert detail.loc["AAPL", "target_quantity"] == 12.0
    assert json.loads(detail.loc["AAPL", "notes"])["target_weight"] == 0.15


def test_get_latest_targets_detail_empty_for_unknown_uid(
    signal_book: SignalBook,
) -> None:
    assert signal_book.get_latest_targets_detail("never-written").empty


def test_later_write_overrides_earlier_target(signal_book: SignalBook) -> None:
    uid = "uid-1"
    signal_book.write_targets(uid, {"AAPL": 10.0}, as_of="2026-01-05", reason="R1")
    signal_book.write_targets(uid, {"AAPL": 12.0}, as_of="2026-02-05", reason="R2")

    assert signal_book.get_latest_targets(uid) == {"AAPL": 12.0}


def test_explicit_zero_target_clears_a_dropped_symbol(
    signal_book: SignalBook,
) -> None:
    uid = "uid-1"
    signal_book.write_targets(
        uid, {"AAPL": 10.0, "MSFT": 5.0}, as_of="2026-01-05", reason="R1"
    )
    signal_book.write_targets(
        uid, {"AAPL": 10.0, "MSFT": 0.0}, as_of="2026-02-05", reason="R2"
    )

    assert signal_book.get_latest_targets(uid) == {"AAPL": 10.0, "MSFT": 0.0}


def test_different_uids_do_not_collide(signal_book: SignalBook) -> None:
    signal_book.write_targets("uid-a", {"AAPL": 10.0}, as_of="2026-01-05", reason="R")
    signal_book.write_targets("uid-b", {"AAPL": 99.0}, as_of="2026-01-05", reason="R")

    assert signal_book.get_latest_targets("uid-a") == {"AAPL": 10.0}
    assert signal_book.get_latest_targets("uid-b") == {"AAPL": 99.0}


def test_get_history_returns_full_audit_trail(signal_book: SignalBook) -> None:
    uid = "uid-1"
    signal_book.write_targets(uid, {"AAPL": 10.0}, as_of="2026-01-05", reason="R1")
    signal_book.write_targets(uid, {"AAPL": 12.0}, as_of="2026-02-05", reason="R2")

    history = signal_book.get_history(uid, symbol="AAPL")
    assert list(history["target_quantity"]) == [10.0, 12.0]
    assert list(history["reason"]) == ["R1", "R2"]


def test_write_targets_with_notes(signal_book: SignalBook) -> None:
    signal_book.write_targets(
        "uid-1",
        {"AAPL": 10.0},
        as_of="2026-01-05",
        reason="MOMENTUM_REBALANCE",
        notes_by_symbol={"AAPL": {"score": 0.42, "rank": 1}},
    )

    history = signal_book.get_history("uid-1")
    assert json.loads(history.iloc[0]["notes"]) == {"score": 0.42, "rank": 1}


def test_write_targets_without_notes_defaults_to_empty_json(
    signal_book: SignalBook,
) -> None:
    signal_book.write_targets(
        "uid-1", {"AAPL": 10.0}, as_of="2026-01-05", reason="R1"
    )

    history = signal_book.get_history("uid-1")
    assert json.loads(history.iloc[0]["notes"]) == {}


def test_empty_targets_is_a_noop(signal_book: SignalBook) -> None:
    signal_book.write_targets("uid-1", {}, as_of="2026-01-05", reason="R")
    assert signal_book.get_latest_targets("uid-1") == {}


def test_get_latest_targets_for_unknown_uid_is_empty(
    signal_book: SignalBook,
) -> None:
    assert signal_book.get_latest_targets("never-written") == {}


def test_list_uids_is_alphabetical_and_deduplicated(signal_book: SignalBook) -> None:
    signal_book.write_targets("uid-b", {"AAPL": 1.0}, as_of="2026-01-05", reason="R")
    signal_book.write_targets("uid-a", {"AAPL": 1.0}, as_of="2026-01-05", reason="R")
    signal_book.write_targets("uid-a", {"AAPL": 2.0}, as_of="2026-02-05", reason="R")

    assert signal_book.list_uids() == ["uid-a", "uid-b"]


def test_list_uids_is_empty_for_a_fresh_book(signal_book: SignalBook) -> None:
    assert signal_book.list_uids() == []


def test_persists_across_reopening_the_same_db_file(tmp_path: Path) -> None:
    db_path = tmp_path / "signal_book.sqlite"

    with SignalBook(db_path=db_path) as book:
        book.write_targets("uid-1", {"AAPL": 10.0}, as_of="2026-01-05", reason="R1")

    with SignalBook(db_path=db_path) as reopened:
        assert reopened.get_latest_targets("uid-1") == {"AAPL": 10.0}
