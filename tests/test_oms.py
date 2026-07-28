from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live.execution.account_state import AccountState, FakeAccountStateProvider
from live.execution.oms import Order, compute_orders, reconcile_uid
from live.execution.signal_book import SignalBook


# ========================================================
# Order
# ========================================================


def test_order_action_is_buy_for_positive_quantity() -> None:
    assert Order(symbol="AAPL", quantity=5.0).action == "BUY"


def test_order_action_is_sell_for_negative_quantity() -> None:
    assert Order(symbol="AAPL", quantity=-5.0).action == "SELL"


# ========================================================
# compute_orders
# ========================================================


def test_new_target_with_no_existing_position_is_a_full_buy() -> None:
    orders = compute_orders(targets={"AAPL": 10.0}, actual_positions={})
    assert orders == [Order(symbol="AAPL", quantity=10.0)]


def test_target_below_actual_generates_a_sell_for_the_difference() -> None:
    orders = compute_orders(targets={"AAPL": 4.0}, actual_positions={"AAPL": 10.0})
    assert orders == [Order(symbol="AAPL", quantity=-6.0)]


def test_target_above_actual_generates_a_buy_for_the_difference() -> None:
    orders = compute_orders(targets={"AAPL": 10.0}, actual_positions={"AAPL": 4.0})
    assert orders == [Order(symbol="AAPL", quantity=6.0)]


def test_explicit_zero_target_generates_a_full_exit() -> None:
    orders = compute_orders(targets={"AAPL": 0.0}, actual_positions={"AAPL": 10.0})
    assert orders == [Order(symbol="AAPL", quantity=-10.0)]


def test_matching_target_and_actual_generates_no_order() -> None:
    orders = compute_orders(targets={"AAPL": 10.0}, actual_positions={"AAPL": 10.0})
    assert orders == []


def test_actual_position_with_no_target_at_all_is_treated_as_implicit_zero() -> None:
    # No signal_book row for MSFT at all (not even an explicit target=0)
    # -- a stray/manual holding this reconciliation still has to close.
    orders = compute_orders(targets={}, actual_positions={"MSFT": 3.0})
    assert orders == [Order(symbol="MSFT", quantity=-3.0)]


def test_deltas_below_min_quantity_are_dropped() -> None:
    orders = compute_orders(
        targets={"AAPL": 10.0000001},
        actual_positions={"AAPL": 10.0},
        min_quantity=1e-3,
    )
    assert orders == []


def test_deltas_at_or_above_min_quantity_are_kept() -> None:
    orders = compute_orders(
        targets={"AAPL": 10.01},
        actual_positions={"AAPL": 10.0},
        min_quantity=1e-3,
    )
    assert orders == [Order(symbol="AAPL", quantity=pytest.approx(0.01))]


def test_multiple_symbols_are_each_diffed_independently_and_sorted() -> None:
    orders = compute_orders(
        targets={"AAPL": 10.0, "MSFT": 0.0, "GOOG": 5.0},
        actual_positions={"AAPL": 4.0, "MSFT": 2.0},
    )
    assert orders == [
        Order(symbol="AAPL", quantity=6.0),
        Order(symbol="GOOG", quantity=5.0),
        Order(symbol="MSFT", quantity=-2.0),
    ]


# ========================================================
# reconcile_uid
# ========================================================


@pytest.fixture
def signal_book(tmp_path: Path):
    with SignalBook(db_path=tmp_path / "signal_book.sqlite") as book:
        yield book


def test_reconcile_uid_composes_signal_book_and_account_state(
    signal_book: SignalBook,
) -> None:
    signal_book.write_targets(
        "uid-1", {"AAPL": 10.0, "MSFT": 0.0}, as_of="2026-01-05", reason="R"
    )

    account_state = AccountState(
        cash=1_000.0,
        positions={"AAPL": 4.0, "MSFT": 2.0},
        equity=1_000.0,
        as_of=pd.Timestamp("2026-01-05"),
    )
    provider = FakeAccountStateProvider(account_state)

    orders = reconcile_uid("uid-1", signal_book, provider)

    assert orders == [
        Order(symbol="AAPL", quantity=6.0),
        Order(symbol="MSFT", quantity=-2.0),
    ]


def test_reconcile_uid_with_no_signal_book_history_treats_all_actual_as_stray(
    signal_book: SignalBook,
) -> None:
    account_state = AccountState(
        cash=1_000.0,
        positions={"AAPL": 4.0},
        equity=1_000.0,
        as_of=pd.Timestamp("2026-01-05"),
    )
    provider = FakeAccountStateProvider(account_state)

    orders = reconcile_uid("never-written-uid", signal_book, provider)

    assert orders == [Order(symbol="AAPL", quantity=-4.0)]


def test_reconcile_uid_with_matching_state_generates_nothing(
    signal_book: SignalBook,
) -> None:
    signal_book.write_targets("uid-1", {"AAPL": 10.0}, as_of="2026-01-05", reason="R")

    account_state = AccountState(
        cash=1_000.0,
        positions={"AAPL": 10.0},
        equity=1_000.0,
        as_of=pd.Timestamp("2026-01-05"),
    )
    provider = FakeAccountStateProvider(account_state)

    assert reconcile_uid("uid-1", signal_book, provider) == []
