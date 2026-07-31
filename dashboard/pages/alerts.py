"""
Atlas -- Alerts page.

A filterable, cross-UID view of run_log -- every scheduled check, not
just failures, since "no run_log entries at all" is itself the thing
worth alerting on (see run_log.py's docstring). Closest thing Atlas has
today to the reference system's service-tagged error log; there's only
one "service" so far (the run_log itself), so this filters by decision
and UID instead of by service name.

The "what happened" column is computed here, not stored anywhere --
run_log's `detail` is empty JSON ({}) for REBALANCED/NO_REBALANCE
decisions today (see live_ems.py's run_live_day()), so a raw dump of
`detail` told you nothing about what a strategy actually did on a given
day. This reconstructs it by diffing signal_book's rows for that
(uid, as_of) against each symbol's immediately preceding target --
cheap and correct because place_trade() (ems.py) already only ever
writes a signal_book row when a symbol's target quantity actually
changed (see EMS.place_trade()'s zero-delta short-circuit), so every
row for a given as_of *is* a real change, never a no-op restatement.
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
run_log = lib.get_run_log(db_path)
signal_book = lib.get_signal_book(db_path)


# ============================================================
# "What happened" summary
# ============================================================

_ZERO = 1e-9


def _signal_book_changes(uid: str) -> pd.DataFrame:
    """Every signal_book row for `uid`, with each row's immediately
    preceding target_quantity for that symbol attached -- lets a single
    as_of's rows be classified as entered/exited/resized without a
    second query per row."""
    history = signal_book.get_history(uid)
    if history.empty:
        return history
    history = history.sort_values("id")
    history["prev_target_quantity"] = history.groupby("symbol")[
        "target_quantity"
    ].shift(1)
    return history


def _classify(prev: float | None, new: float) -> str:
    if prev is None or pd.isna(prev) or abs(prev) <= _ZERO:
        return "entered"
    if abs(new) <= _ZERO:
        return "exited"
    return "resized"


def _rebalance_summary(changes_by_uid: dict[str, pd.DataFrame], uid: str, as_of: str) -> str:
    if uid not in changes_by_uid:
        changes_by_uid[uid] = _signal_book_changes(uid)
    history = changes_by_uid[uid]
    if history.empty:
        return "no signal book rows found"

    day_rows = history[history["as_of"] == as_of]
    if day_rows.empty:
        return "no signal book rows for this as_of"

    kinds = [
        _classify(prev, new)
        for prev, new in zip(day_rows["prev_target_quantity"], day_rows["target_quantity"])
    ]
    day_rows = day_rows.assign(kind=kinds)

    entered = sorted(day_rows.loc[day_rows["kind"] == "entered", "symbol"])
    exited = sorted(day_rows.loc[day_rows["kind"] == "exited", "symbol"])
    resized = int((day_rows["kind"] == "resized").sum())

    parts = []
    if entered:
        parts.append(f"+{len(entered)} in ({', '.join(entered)})")
    if exited:
        parts.append(f"-{len(exited)} out ({', '.join(exited)})")
    if resized:
        parts.append(f"{resized} reweighted")
    return " · ".join(parts) if parts else "no change"


def _detail_dict(detail_json: str | None) -> dict:
    try:
        return json.loads(detail_json) if detail_json else {}
    except (TypeError, ValueError):
        return {}


def _data_refresh_summary(detail: dict) -> str | None:
    """market_data_refresh's detail already carries a real summary
    (rows_processed/successful/failed) -- just format it, don't
    recompute anything."""
    if "rows_processed" not in detail:
        return None
    pieces = [f"{detail.get('rows_processed', 0):,} rows"]
    successful, requested = detail.get("successful"), detail.get("symbols_requested")
    if successful is not None and requested is not None:
        pieces.append(f"{successful}/{requested} symbols OK")
    failed = detail.get("failed") or []
    if failed:
        shown = ", ".join(failed[:5]) + ("…" if len(failed) > 5 else "")
        pieces.append(f"failed: {shown}")
    return ", ".join(pieces)


def _summarize(row: pd.Series, changes_by_uid: dict[str, pd.DataFrame]) -> str:
    detail = _detail_dict(row["detail"])

    if row["decision"] == "ERROR" and "error" in detail:
        return f"ERROR: {detail['error']}"

    data_refresh_summary = _data_refresh_summary(detail)
    if data_refresh_summary is not None:
        return data_refresh_summary

    if row["decision"] == "REBALANCED":
        return _rebalance_summary(changes_by_uid, row["uid"], row["as_of"])
    if row["decision"] == "NO_REBALANCE":
        return "no change"
    if row["decision"] == "KILLED":
        return "skipped (kill switch)"

    # Forward-compatible fallback for any decision/detail shape this
    # page doesn't specifically know about yet -- same "don't need page
    # changes for new decision values" principle the decision tabs use.
    return row["detail"] if row["detail"] and row["detail"] != "{}" else "—"

st.title("Alerts")
st.caption("Every recorded check across every UID, regardless of outcome.")

if st.sidebar.button("Refresh"):
    st.rerun()

all_history = run_log.get_all_history()

if all_history.empty:
    st.info(
        "No checks recorded yet at this DB path. Run "
        "`live.execution.live_ems.run_live_day(..., run_log=...)` for a "
        "UID and refresh this page."
    )
    st.stop()

decision_counts = all_history["decision"].value_counts()

decision_options = ["All"] + sorted(all_history["decision"].unique())
tabs = st.columns(len(decision_options))
selected_decision = st.session_state.get("alerts_decision_filter", "All")

for col, option in zip(tabs, decision_options):
    label = option
    if option != "All":
        label = f"{option} ({int(decision_counts.get(option, 0))})"
    else:
        label = f"All ({len(all_history)})"
    if col.button(label, key=f"decision-tab-{option}"):
        st.session_state["alerts_decision_filter"] = option
        st.rerun()

selected_decision = st.session_state.get("alerts_decision_filter", "All")
st.caption(f"Showing: {selected_decision}")

uid_options = ["All"] + sorted(all_history["uid"].unique())
selected_uid = st.selectbox("UID", uid_options)

text_filter = st.text_input("Filter message (searches the detail column)")

filtered = all_history
if selected_decision != "All":
    filtered = filtered[filtered["decision"] == selected_decision]
if selected_uid != "All":
    filtered = filtered[filtered["uid"] == selected_uid]
if text_filter:
    filtered = filtered[
        filtered["detail"].str.contains(text_filter, case=False, na=False)
    ]

changes_by_uid: dict[str, pd.DataFrame] = {}
filtered = filtered.assign(
    what_happened=filtered.apply(lambda row: _summarize(row, changes_by_uid), axis=1)
)

st.dataframe(
    filtered[["ran_at", "uid", "as_of", "decision", "what_happened", "data_as_of", "detail"]],
    width="stretch",
    hide_index=True,
    column_config={
        "decision": st.column_config.TextColumn("decision", width="small"),
        "what_happened": st.column_config.TextColumn("what happened", width="large"),
        "detail": st.column_config.TextColumn("raw detail", width="small"),
    },
)

st.caption(f"{len(filtered)} of {len(all_history)} recorded checks shown.")
