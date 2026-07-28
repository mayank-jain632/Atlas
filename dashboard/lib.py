"""
Shared helpers for every dashboard page: cached DB-backed resources and
the DB-path session-state wiring, so each page under pages/ doesn't
reimplement it. Importing this module also puts the Atlas project root
on sys.path, which is what makes `from live.execution... import ...`
work from any page.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from live.execution.kill_switch import KillSwitch
from live.execution.run_log import RunLog
from live.execution.signal_book import DEFAULT_SIGNAL_BOOK_PATH, SignalBook

DB_PATH_KEY = "db_path"


def init_db_path() -> str:
    """Ensure a valid db_path exists in session_state (shared across
    every page), defaulting to the signal_book/run_log/kill_switch
    default, and return it.

    Checks for falsy (not just missing), not just `not in
    session_state`: once the sidebar text_input widget in app.py has
    rendered once, it owns this key going forward, so if it's ever
    cleared to an empty string (browser glitch, accidental backspace,
    whatever), a mere presence check would never reset it -- and
    Path("") resolves to ".", which SQLite correctly refuses to open as
    a database file, producing a confusing "unable to open database
    file" error far from its actual cause. Resetting on any falsy value
    is what actually prevents that from sticking.
    """
    if not st.session_state.get(DB_PATH_KEY):
        st.session_state[DB_PATH_KEY] = str(DEFAULT_SIGNAL_BOOK_PATH)
    return st.session_state[DB_PATH_KEY]


def _require_valid_db_path(db_path: str) -> None:
    if not db_path or not db_path.strip():
        raise ValueError(
            "DB path is empty. Check the sidebar 'Signal book / run "
            "log / kill switch DB path' field, or clear it and refresh "
            "to fall back to the default."
        )


@st.cache_resource
def get_signal_book(db_path: str) -> SignalBook:
    _require_valid_db_path(db_path)
    return SignalBook(db_path=db_path)


@st.cache_resource
def get_run_log(db_path: str) -> RunLog:
    _require_valid_db_path(db_path)
    return RunLog(db_path=db_path)


@st.cache_resource
def get_kill_switch(db_path: str) -> KillSwitch:
    _require_valid_db_path(db_path)
    return KillSwitch(db_path=db_path)
