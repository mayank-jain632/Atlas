"""Atlas -- Live Data page. Placeholder: no live/streaming data connection exists yet."""

from __future__ import annotations

from pathlib import Path
import sys

_DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import streamlit as st

st.title("Live Data")
st.info(
    "Not built yet, and not the same shape as the reference system's "
    "version even once it is: Atlas trades equities/futures on a daily "
    "or hourly cadence, not 0DTE options on a per-minute cadence, so "
    "there's no option chain here and (per an earlier decision) no "
    "persistent streaming subscription -- `IBKRClient.get_last_price(s)` "
    "(live/execution/ibkr_client.py) already exists for one-shot polled "
    "snapshots, it's just not wired into anything that runs continuously "
    "yet. This page will show polled price checks once it is, not "
    "live tick-by-tick candles."
)
