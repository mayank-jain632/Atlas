from __future__ import annotations

import pandas as pd

from live.execution.account_state import AccountState, FakeAccountStateProvider


def _sample_state() -> AccountState:
    return AccountState(
        cash=5_000.0,
        positions={"IBM": 10.0, "AAPL": 2.0},
        equity=7_500.0,
        as_of=pd.Timestamp("2026-07-15"),
    )


def test_account_state_holds_its_fields() -> None:
    state = _sample_state()

    assert state.cash == 5_000.0
    assert state.positions == {"IBM": 10.0, "AAPL": 2.0}
    assert state.equity == 7_500.0
    assert state.as_of == pd.Timestamp("2026-07-15")


def test_fake_provider_returns_the_state_it_was_built_with() -> None:
    state = _sample_state()
    provider = FakeAccountStateProvider(account_state=state)

    assert provider.get_account_state() is state


def test_fake_provider_returns_the_same_state_on_repeated_calls() -> None:
    provider = FakeAccountStateProvider(account_state=_sample_state())

    first = provider.get_account_state()
    second = provider.get_account_state()

    assert first == second
