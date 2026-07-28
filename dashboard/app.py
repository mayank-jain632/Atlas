"""
Atlas -- live execution dashboard entrypoint.

Multi-page structure, sidebar navigation. Pages with real data behind
them today: System Health, Signals, Alerts. OMS / Live Data / Live vs BT
are placeholders -- Atlas has no OMS, no live IBKR connection wired up
anywhere, and no live-vs-backtest comparison job yet, so those pages say
so rather than pretending to show something real.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

_DASHBOARD_DIR = Path(__file__).resolve().parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import streamlit as st

import lib

st.set_page_config(page_title="Atlas", layout="wide")

lib.init_db_path()

st.sidebar.title("Atlas")
st.sidebar.caption("Live execution")
st.sidebar.text_input(
    "Signal book / run log / kill switch DB path",
    key=lib.DB_PATH_KEY,
    help="Shared across every page -- change it once here, not per page.",
)

pages = [
    st.Page(
        "pages/system_health.py",
        title="System Health",
        icon=":material/monitor_heart:",
        default=True,
    ),
    st.Page("pages/signals.py", title="Signals", icon=":material/podcasts:"),
    st.Page("pages/alerts.py", title="Alerts", icon=":material/warning:"),
    st.Page("pages/data.py", title="Data", icon=":material/database:"),
    st.Page("pages/oms.py", title="OMS", icon=":material/receipt_long:"),
    st.Page("pages/live_data.py", title="Live Data", icon=":material/show_chart:"),
    st.Page(
        "pages/live_vs_backtest.py",
        title="Live vs BT",
        icon=":material/compare_arrows:",
    ),
]

navigation = st.navigation(pages)
navigation.run()
