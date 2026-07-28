"""
Regression tests for the dashboard, run headlessly via Streamlit's
official AppTest API (streamlit.testing.v1) -- no browser needed.

Covers two real bugs found by exercising interactions rather than just
page loads:
1. SQLite connections cached via st.cache_resource must tolerate being
   used from a different thread than the one that created them (a
   Streamlit rerun, e.g. after a button click, can land on a different
   thread) -- see check_same_thread=False in signal_book.py/run_log.py/
   kill_switch.py.
2. dashboard/lib.py's init_db_path() must recover from an empty (not
   just missing) session_state value -- once the sidebar text_input has
   rendered once, it owns that key, so a mere "not in session_state"
   check never resets an emptied value, and Path("") silently resolves
   to "." rather than raising anything obviously related to the cause.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

import live.execution.alerting as alerting

PAGES = [
    "dashboard/app.py",
    "dashboard/pages/system_health.py",
    "dashboard/pages/signals.py",
    "dashboard/pages/alerts.py",
    "dashboard/pages/data.py",
    "dashboard/pages/oms.py",
    "dashboard/pages/live_data.py",
    "dashboard/pages/live_vs_backtest.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_loads_without_exceptions(page: str) -> None:
    at = AppTest.from_file(page)
    at.run(timeout=30)
    assert not at.exception


@pytest.mark.parametrize(
    "page",
    [
        "dashboard/pages/system_health.py",
        "dashboard/pages/signals.py",
        "dashboard/pages/alerts.py",
    ],
)
def test_empty_db_path_in_session_state_recovers_to_default(page: str) -> None:
    at = AppTest.from_file(page)
    at.session_state["db_path"] = ""
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["db_path"], "db_path should have been reset, not left empty"


def test_kill_switch_toggle_survives_a_rerun_on_a_different_thread(
    tmp_path: Path,
) -> None:
    """Regression test for the check_same_thread bug: toggling a kill
    switch triggers a rerun via st.rerun(), which is exactly the
    button-click-causes-a-rerun-on-a-new-thread scenario that broke
    before check_same_thread=False was added. Uses its own DB so it
    doesn't mutate the real default-path demo data."""
    at = AppTest.from_file("dashboard/pages/system_health.py")
    at.session_state["db_path"] = str(tmp_path / "system_health_test.sqlite")
    at.run(timeout=30)

    kill_button = next(b for b in at.button if b.label == "Kill entire system")
    kill_button.click().run(timeout=30)
    assert not at.exception
    assert any(e.value == "SYSTEM KILLED" for e in at.error)

    unkill_button = next(b for b in at.button if b.label == "Un-kill system")
    unkill_button.click().run(timeout=30)
    assert not at.exception
    assert any(s.value == "System live" for s in at.success)


def test_alerting_switch_appears_alongside_the_other_process_switches(
    tmp_path: Path,
) -> None:
    """"alerting" is just another DEFAULT_SWITCHES entry, so it should
    show up in the process-switch row without any page-specific code
    for it -- proves the generic UI actually picked it up."""
    at = AppTest.from_file("dashboard/pages/system_health.py")
    at.session_state["db_path"] = str(tmp_path / "system_health_test.sqlite")
    at.run(timeout=30)

    assert not at.exception
    assert any("Alerting" in md.value for md in at.markdown)
    assert any(b.key == "kill-alerting" for b in at.button)


def test_killing_a_switch_notifies_via_the_configured_notifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the wiring, not a real Telegram send: notifier_from_env()
    is replaced with a fake that records what it was asked to send."""
    sent: list[tuple[str, bool, str]] = []

    class _FakeNotifier:
        def notify_kill_switch(self, name: str, killed: bool, reason: str = "") -> bool:
            sent.append((name, killed, reason))
            return True

    monkeypatch.setattr(alerting, "notifier_from_env", lambda kill_switch: _FakeNotifier())

    at = AppTest.from_file("dashboard/pages/system_health.py")
    at.session_state["db_path"] = str(tmp_path / "system_health_test.sqlite")
    at.run(timeout=30)

    kill_button = next(b for b in at.button if b.label == "Kill entire system")
    kill_button.click().run(timeout=30)

    assert not at.exception
    assert len(sent) == 1
    assert sent[0][0] == "system"
    assert sent[0][1] is True
