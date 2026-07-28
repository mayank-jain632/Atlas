"""Atlas -- Live vs BT page. Placeholder: no live-vs-backtest comparison job exists yet."""

from __future__ import annotations

from pathlib import Path
import sys

_DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import streamlit as st

st.title("Live vs BT")
st.info(
    "Not built yet. The plan (from the live-execution architecture "
    "discussion) is that this should be cheap to build once it's "
    "needed: live execution reuses the exact same strategy classes as "
    "backtesting (no separate live-runner implementation), so any "
    "divergence between a day's real signal-book output and a backtest "
    "replay of that same day can only come from data or timing, not "
    "diverging logic. This page will show that diff, per UID, once "
    "there's enough live history to make it meaningful."
)
