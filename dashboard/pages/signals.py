"""
Atlas -- Signals page.

For one chosen UID: whether the scheduled signal check is running on
time, what it decided, and what it currently wants to hold. This is
the page that proves signals fire correctly before anything places a
real (even paper) order -- see live/execution/run_log.py's docstring
for why the run log is what makes this provable at all.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

_DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import pandas as pd
import streamlit as st

import lib

db_path = lib.init_db_path()
signal_book = lib.get_signal_book(db_path)
run_log = lib.get_run_log(db_path)

known_uids = sorted(set(signal_book.list_uids()) | set(run_log.list_uids()))

if not known_uids:
    st.title("Signals")
    st.info(
        "No UIDs recorded yet at this DB path. Run "
        "`live.execution.live_ems.run_live_day(..., run_log=...)` for a "
        "UID and refresh this page."
    )
    st.stop()

selected_uid = st.sidebar.selectbox("UID", known_uids)

stale_after_hours = st.sidebar.number_input(
    "Flag as stale after (hours)",
    min_value=1,
    value=26,
    help=(
        "A daily check should fire roughly once every 24h. Flag it if "
        "the last recorded run is older than this."
    ),
)

if st.sidebar.button("Refresh"):
    st.rerun()


# ============================================================
# Header + staleness check
# ============================================================

st.title("Signals")
st.caption(selected_uid)

latest_run = run_log.get_latest(selected_uid)

status_col, as_of_col, data_col, staleness_col = st.columns(4)

if latest_run is None:
    status_col.metric("Last decision", "no run log entry yet")
else:
    decision = latest_run["decision"]
    status_col.metric("Last decision", decision)
    as_of_col.metric("As of (trading date)", str(latest_run["as_of"])[:10])
    data_col.metric(
        "Data as of",
        str(latest_run["data_as_of"])[:19] if latest_run["data_as_of"] else "—",
    )

    ran_at = pd.Timestamp(latest_run["ran_at"])
    age_hours = (pd.Timestamp.now() - ran_at).total_seconds() / 3600.0
    staleness_col.metric("Hours since last check", f"{age_hours:.1f}")

    if decision == "ERROR":
        st.error(f"Last check errored: {latest_run['detail']}")
    elif age_hours > stale_after_hours:
        st.warning(
            f"Last check was {age_hours:.1f}h ago, more than the "
            f"{stale_after_hours:.0f}h threshold — the scheduled "
            "check may not be running."
        )
    else:
        st.success("Running on schedule.")


# ============================================================
# Current targets
# ============================================================

st.subheader("Current target portfolio")

targets_detail = signal_book.get_latest_targets_detail(selected_uid)
targets_detail = targets_detail[targets_detail["target_quantity"].abs() > 1e-9]

if targets_detail.empty:
    st.write("No open targets.")
else:
    def _from_notes(notes_json: str, field: str) -> float | None:
        try:
            return json.loads(notes_json).get(field)
        except (TypeError, ValueError, AttributeError):
            return None

    targets_df = targets_detail[["symbol", "target_quantity"]].copy()
    targets_df["target_weight"] = targets_detail["notes"].apply(
        lambda n: _from_notes(n, "target_weight")
    )
    targets_df["score"] = targets_detail["notes"].apply(
        lambda n: _from_notes(n, "score")
    )

    has_weight = targets_df["target_weight"].notna().any()
    sort_column = "target_weight" if has_weight else "target_quantity"
    targets_df = targets_df.sort_values(sort_column, ascending=False).reset_index(
        drop=True
    )

    left, right = st.columns([1, 1])
    left.dataframe(
        targets_df,
        width="stretch",
        hide_index=True,
        column_config={
            "target_quantity": st.column_config.NumberColumn(
                "shares", format="%.2f"
            ),
            "target_weight": st.column_config.NumberColumn(
                "target weight", format="percent"
            ),
            "score": st.column_config.NumberColumn("score", format="%.3f"),
        },
    )

    if has_weight:
        right.bar_chart(targets_df.set_index("symbol")["target_weight"])
        right.caption(
            "Target weight (% of portfolio) — comparable across symbols "
            "regardless of share price, unlike raw share count."
        )
    else:
        right.bar_chart(targets_df.set_index("symbol")["target_quantity"])
        right.caption(
            "Falling back to raw share count: this UID's signal book "
            "notes don't include target_weight."
        )


# ============================================================
# History
# ============================================================

st.subheader("Run history")

run_history = run_log.get_history(selected_uid, limit=100)
st.dataframe(
    run_history[["ran_at", "as_of", "data_as_of", "decision", "detail"]],
    width="stretch",
    hide_index=True,
)

with st.expander("Full signal book history"):
    signal_history = signal_book.get_history(selected_uid)
    st.dataframe(signal_history, width="stretch", hide_index=True)
