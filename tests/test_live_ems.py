"""
End-to-end signal test: run a real, unmodified momentum strategy for one
real day against the existing market_data.duckdb, and check that its
decisions land correctly in the signal book. No IBKR, no scheduler --
just the two pieces built so far (SignalBookLiveMixin + SignalBook)
wired to a real strategy class.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live.execution.account_state import AccountState, FakeAccountStateProvider
from live.execution.live_ems import make_live_strategy, run_live_day
from live.execution.run_log import RunLog
from live.execution.signal_book import SignalBook
from strategies.momentum.momentum import MomentumStrategy


DB_PATH = PROJECT_ROOT / "duckdb" / "market_data.duckdb"
UID = "momentum__u=nasdaq100__sig=price__lb=90__rb=monthly__n=10__alloc=score"
AS_OF = "2026-07-15"  # last date available in market_data.duckdb at write time


UNIVERSE_ROOT = str(PROJECT_ROOT / "config" / "universes")


def _build_live_strategy(signal_book: SignalBook):
    return make_live_strategy(
        MomentumStrategy,
        signal_book=signal_book,
        uid=UID,
        capital=100_000.0,
        db_path=DB_PATH,
        timeframe="1d",
        allow_fractional_shares=True,
        universe_root=UNIVERSE_ROOT,
    )


@pytest.fixture
def signal_book(tmp_path: Path):
    with SignalBook(db_path=tmp_path / "signal_book.sqlite") as book:
        yield book


@pytest.fixture
def live_strategy(signal_book: SignalBook):
    if not DB_PATH.exists():
        pytest.skip(f"{DB_PATH} not available on this machine")

    return _build_live_strategy(signal_book)


def test_first_run_rebalances_and_writes_to_signal_book(
    live_strategy,
    signal_book: SignalBook,
) -> None:
    # A freshly constructed strategy has never rebalanced
    # (last_rebalance_key is None), so should_rebalance() fires on the
    # very first day it's given, same as it would in a backtest.
    run_live_day(live_strategy, as_of=AS_OF)

    targets = signal_book.get_latest_targets(UID)

    assert targets, "expected the strategy to have rebalanced and written targets"
    assert len(targets) <= int(live_strategy.parameters["top_n"])
    assert all(quantity > 0 for quantity in targets.values()), (
        "momentum (long-only, no allow_short_positions) should never "
        "target a negative quantity"
    )

    # The signal book's record should agree with the strategy's own
    # simulated ledger -- same source of truth, just two places it's
    # visible.
    assert set(targets) == set(live_strategy.get_positions())
    for symbol, quantity in targets.items():
        assert quantity == pytest.approx(live_strategy.get_position(symbol))


def test_second_call_same_day_is_a_noop(
    live_strategy,
    signal_book: SignalBook,
) -> None:
    run_live_day(live_strategy, as_of=AS_OF)
    first_history_len = len(signal_book.get_history(UID))

    # should_rebalance() only fires once per rebalance period; calling
    # again for the same day must not double-write.
    run_live_day(live_strategy, as_of=AS_OF)

    assert len(signal_book.get_history(UID)) == first_history_len


def test_signal_book_notes_carry_score_and_rank(
    live_strategy,
    signal_book: SignalBook,
) -> None:
    import json

    run_live_day(live_strategy, as_of=AS_OF)

    history = signal_book.get_history(UID)
    notes = json.loads(history.iloc[0]["notes"])

    assert "score" in notes
    assert "rank" in notes


def test_seeding_with_limited_cash_scales_targets_proportionally(
    tmp_path: Path,
) -> None:
    if not DB_PATH.exists():
        pytest.skip(f"{DB_PATH} not available on this machine")

    # Baseline: flat $100,000 capital, empty book (no seeding) -- the
    # existing behavior this whole file already exercises.
    with SignalBook(db_path=tmp_path / "unseeded.sqlite") as unseeded_book:
        run_live_day(_build_live_strategy(unseeded_book), as_of=AS_OF)
        baseline_targets = unseeded_book.get_latest_targets(UID)

    # Same UID, same day, but the account only actually has $5,000 cash
    # and no existing positions.
    with SignalBook(db_path=tmp_path / "seeded.sqlite") as seeded_book:
        account_state = AccountState(
            cash=5_000.0,
            positions={},
            equity=5_000.0,
            as_of=pd.Timestamp(AS_OF),
        )
        run_live_day(
            _build_live_strategy(seeded_book),
            as_of=AS_OF,
            account_state_provider=FakeAccountStateProvider(account_state),
        )
        seeded_targets = seeded_book.get_latest_targets(UID)

    assert baseline_targets, "expected the baseline run to produce targets"
    assert set(seeded_targets) == set(baseline_targets), (
        "seeding less cash shouldn't change which stocks get picked "
        "(selection is score-based, not capital-based), only how much "
        "of each"
    )

    scale = 5_000.0 / 100_000.0
    for symbol, baseline_quantity in baseline_targets.items():
        assert seeded_targets[symbol] == pytest.approx(
            baseline_quantity * scale, rel=1e-6
        )


def test_seeding_a_stale_holding_produces_an_explicit_exit(
    tmp_path: Path,
) -> None:
    if not DB_PATH.exists():
        pytest.skip(f"{DB_PATH} not available on this machine")

    # Learn today's actual picks first, so the holding seeded below is
    # guaranteed not to be one of them -- not hardcoded, so this doesn't
    # go stale if the underlying price data changes later.
    with SignalBook(db_path=tmp_path / "baseline.sqlite") as baseline_book:
        baseline_strategy = _build_live_strategy(baseline_book)
        run_live_day(baseline_strategy, as_of=AS_OF)
        picked_today = set(baseline_book.get_latest_targets(UID))

    stale_holding = next(
        symbol
        for symbol in baseline_strategy.stock_universe
        if symbol not in picked_today
    )

    with SignalBook(db_path=tmp_path / "seeded.sqlite") as seeded_book:
        account_state = AccountState(
            cash=1_000.0,
            positions={stale_holding: 5.0},
            equity=1_000.0,
            as_of=pd.Timestamp(AS_OF),
        )
        run_live_day(
            _build_live_strategy(seeded_book),
            as_of=AS_OF,
            account_state_provider=FakeAccountStateProvider(account_state),
        )
        seeded_targets = seeded_book.get_latest_targets(UID)

    assert stale_holding in seeded_targets, (
        f"expected an explicit exit row for {stale_holding}, which was "
        "seeded as held but isn't one of today's picks"
    )
    assert seeded_targets[stale_holding] == 0.0


def test_run_log_distinguishes_rebalanced_from_no_rebalance(
    live_strategy,
    tmp_path: Path,
) -> None:
    with RunLog(db_path=tmp_path / "run_log.sqlite") as log:
        run_live_day(live_strategy, as_of=AS_OF, run_log=log)
        first_entry = log.get_latest(UID)
        assert first_entry is not None
        assert first_entry["decision"] == "REBALANCED"

        # should_rebalance() only fires once per period, so calling
        # again for the same day should log distinctly -- this is the
        # exact case the run log exists for: proving the check ran a
        # second time even though there was nothing new to do.
        run_live_day(live_strategy, as_of=AS_OF, run_log=log)
        second_entry = log.get_latest(UID)
        assert second_entry["decision"] == "NO_REBALANCE"

        assert len(log.get_history(UID)) == 2


class _ExplodingStrategy:
    """Minimal stand-in for a strategy whose on_day_close() blows up --
    run_live_day() only needs these three methods, so a real
    MomentumStrategy isn't necessary to test the error path."""

    uid = "exploding-uid"

    def set_current_timestamp(self, as_of) -> None:
        pass

    def on_day_close(self) -> None:
        raise RuntimeError("boom")

    def flush_signal_book(self) -> bool:
        return False  # pragma: no cover -- unreachable, on_day_close raises first


def test_run_log_records_error_and_reraises(tmp_path: Path) -> None:
    with RunLog(db_path=tmp_path / "run_log.sqlite") as log:
        with pytest.raises(RuntimeError, match="boom"):
            run_live_day(_ExplodingStrategy(), as_of=AS_OF, run_log=log)

        entry = log.get_latest("exploding-uid")
        assert entry is not None
        assert entry["decision"] == "ERROR"
        assert "boom" in entry["detail"]
