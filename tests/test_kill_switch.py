from __future__ import annotations

from pathlib import Path

import pytest

from live.execution.kill_switch import DEFAULT_SWITCHES, SYSTEM_SWITCH, KillSwitch


@pytest.fixture
def kill_switch(tmp_path: Path):
    with KillSwitch(db_path=tmp_path / "kill_switch.sqlite") as ks:
        yield ks


def test_defaults_are_seeded_and_not_killed(kill_switch: KillSwitch) -> None:
    for name in DEFAULT_SWITCHES:
        assert kill_switch.is_killed(name) is False

    all_switches = kill_switch.get_all()
    assert set(all_switches["name"]) == set(DEFAULT_SWITCHES)
    assert (all_switches["killed"] == 0).all()


def test_set_killed_and_is_killed(kill_switch: KillSwitch) -> None:
    kill_switch.set_killed("oms_trader", True, reason="manual test")

    assert kill_switch.is_killed("oms_trader") is True
    assert kill_switch.is_killed("market_data") is False


def test_unset_killed_switch_can_be_turned_back_off(kill_switch: KillSwitch) -> None:
    kill_switch.set_killed("oms_trader", True)
    assert kill_switch.is_killed("oms_trader") is True

    kill_switch.set_killed("oms_trader", False)
    assert kill_switch.is_killed("oms_trader") is False


def test_system_switch_kills_everything(kill_switch: KillSwitch) -> None:
    kill_switch.set_killed(SYSTEM_SWITCH, True, reason="emergency stop")

    assert kill_switch.is_killed(SYSTEM_SWITCH) is True
    assert kill_switch.is_killed("market_data") is True
    assert kill_switch.is_killed("strategy_engine") is True
    assert kill_switch.is_killed("some-uid-never-seen-before") is True


def test_system_switch_off_does_not_affect_others(kill_switch: KillSwitch) -> None:
    kill_switch.set_killed("oms_trader", True)
    assert kill_switch.is_killed(SYSTEM_SWITCH) is False
    assert kill_switch.is_killed("market_data") is False
    assert kill_switch.is_killed("oms_trader") is True


def test_per_uid_switch_is_created_on_first_use(kill_switch: KillSwitch) -> None:
    uid = "momentum__u=nasdaq100__sig=price__lb=90__rb=monthly__n=10__alloc=score"

    assert kill_switch.is_killed(uid) is False

    kill_switch.set_killed(uid, True, reason="pausing this UID")
    assert kill_switch.is_killed(uid) is True

    all_switches = kill_switch.get_all()
    assert uid in set(all_switches["name"])


def test_get_all_reflects_reason_and_updated_by(kill_switch: KillSwitch) -> None:
    kill_switch.set_killed(
        "oms_trader", True, reason="connectivity issue", updated_by="rishab"
    )

    row = kill_switch.get_all().set_index("name").loc["oms_trader"]
    assert row["reason"] == "connectivity issue"
    assert row["updated_by"] == "rishab"


def test_persists_across_reopening_the_same_db_file(tmp_path: Path) -> None:
    db_path = tmp_path / "kill_switch.sqlite"

    with KillSwitch(db_path=db_path) as ks:
        ks.set_killed("strategy_engine", True, reason="testing persistence")

    with KillSwitch(db_path=db_path) as reopened:
        assert reopened.is_killed("strategy_engine") is True
