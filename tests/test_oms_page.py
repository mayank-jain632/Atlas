"""
Regression tests for the OMS dashboard page, run headlessly via
AppTest. Streamlit's AppTest API (as of the version pinned in
requirements.txt) has no interaction support for st.data_editor, so
these tests can't simulate editing the "assumed actual positions"
table -- only its default state (zero for every targeted symbol) is
exercised here. The reconciliation logic itself (including non-zero
actual positions, stray positions, etc.) is already covered directly
in tests/test_oms.py without going through the UI at all.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live.execution.signal_book import SignalBook

PAGE = "dashboard/pages/oms.py"


@pytest.fixture
def signal_book_path(tmp_path: Path) -> Path:
    return tmp_path / "signal_book.sqlite"


def test_default_zero_actual_positions_yields_full_buy_orders(
    signal_book_path: Path,
) -> None:
    with SignalBook(db_path=signal_book_path) as book:
        book.write_targets(
            "uid-1", {"AAPL": 10.0, "MSFT": 5.0}, as_of="2026-01-05", reason="R"
        )

    at = AppTest.from_file(PAGE)
    at.session_state["db_path"] = str(signal_book_path)
    at.run(timeout=30)

    assert not at.exception

    orders_table = at.dataframe[-1].value
    assert set(orders_table["symbol"]) == {"AAPL", "MSFT"}
    assert (orders_table["action"] == "BUY").all()
    assert dict(zip(orders_table["symbol"], orders_table["quantity"])) == {
        "AAPL": 10.0,
        "MSFT": 5.0,
    }


def test_target_positions_table_excludes_explicit_zero_targets(
    signal_book_path: Path,
) -> None:
    with SignalBook(db_path=signal_book_path) as book:
        book.write_targets(
            "uid-1", {"AAPL": 10.0, "MSFT": 0.0}, as_of="2026-01-05", reason="R"
        )

    at = AppTest.from_file(PAGE)
    at.session_state["db_path"] = str(signal_book_path)
    at.run(timeout=30)

    assert not at.exception

    targets_table = at.dataframe[0].value
    assert list(targets_table["symbol"]) == ["AAPL"]


def test_no_uids_shows_an_info_message_not_an_exception(
    signal_book_path: Path,
) -> None:
    # An empty signal book (never written to) is still a valid state --
    # just nothing to reconcile yet.
    with SignalBook(db_path=signal_book_path):
        pass

    at = AppTest.from_file(PAGE)
    at.session_state["db_path"] = str(signal_book_path)
    at.run(timeout=30)

    assert not at.exception
    assert any("No UIDs recorded yet" in info.value for info in at.info)
