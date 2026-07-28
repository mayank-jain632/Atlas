"""
Atlas -- Alerts page.

A filterable, cross-UID view of run_log -- every scheduled check, not
just failures, since "no run_log entries at all" is itself the thing
worth alerting on (see run_log.py's docstring). Closest thing Atlas has
today to the reference system's service-tagged error log; there's only
one "service" so far (the run_log itself), so this filters by decision
and UID instead of by service name.
"""

from __future__ import annotations

from pathlib import Path
import sys

_DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import streamlit as st

import lib

db_path = lib.init_db_path()
run_log = lib.get_run_log(db_path)

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

st.dataframe(
    filtered[["ran_at", "uid", "as_of", "data_as_of", "decision", "detail"]],
    width="stretch",
    hide_index=True,
    column_config={
        "decision": st.column_config.TextColumn("decision", width="small"),
    },
)

st.caption(f"{len(filtered)} of {len(all_history)} recorded checks shown.")
