"""
Atlas -- System Health page.

Two things: a per-UID run-health grid across every UID at once (the
Signals page only shows one UID at a time), and the kill switches --
system-wide, named processes, and per-UID. This is the dashboard-
editable source of truth live/run_daily_check.py and
live/run_data_refresh.py actually check before running each time they
fire; toggling one here also pushes a Telegram alert (see
live/execution/alerting.py) if TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are
configured -- unset by default, so this degrades to dashboard-only
until they are.
"""

from __future__ import annotations

from pathlib import Path
import sys

_DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import pandas as pd
import streamlit as st

import lib
from live.execution.alerting import notifier_from_env
from live.execution.kill_switch import DEFAULT_SWITCHES, SYSTEM_SWITCH

db_path = lib.init_db_path()
signal_book = lib.get_signal_book(db_path)
run_log = lib.get_run_log(db_path)
kill_switch = lib.get_kill_switch(db_path)
notifier = notifier_from_env(kill_switch)

st.title("System Health")

if st.sidebar.button("Refresh"):
    st.rerun()

stale_after_hours = st.sidebar.number_input(
    "Flag as stale after (hours)",
    min_value=1,
    value=26,
)


def _toggle_kill(name: str, killed: bool, reason: str = "") -> None:
    """Flip a switch and alert about it in one place, so every kill/
    unkill button below stays a one-line call instead of repeating the
    set_killed()+notify pairing three times."""
    kill_switch.set_killed(name, killed, reason=reason or "manual, from dashboard")
    if notifier is not None:
        notifier.notify_kill_switch(name, killed, reason=reason)


# ============================================================
# Kill switches
# ============================================================

st.subheader("Kill switches")
st.caption(
    "Per-UID pauses that UID's signals · a named process switch would "
    "stop that process (once one exists) · system stops everything."
)

system_killed = kill_switch.is_killed(SYSTEM_SWITCH)

system_col, *_ = st.columns([2, 3, 3, 3])
with system_col:
    if system_killed:
        st.error("🔴 SYSTEM KILLED")
        if st.button("Un-kill system", type="secondary"):
            _toggle_kill(SYSTEM_SWITCH, False)
            st.rerun()
    else:
        st.success("🟢 System live")
        if st.button("Kill entire system", type="primary"):
            _toggle_kill(SYSTEM_SWITCH, True)
            st.rerun()

process_names = [name for name in DEFAULT_SWITCHES if name != SYSTEM_SWITCH]
process_cols = st.columns(len(process_names))

for col, name in zip(process_cols, process_names):
    with col:
        is_killed = kill_switch.is_killed(name)
        st.markdown(f"**{name.replace('_', ' ').title()}**")
        if system_killed:
            st.caption("killed via system switch")
        elif is_killed:
            st.caption("🔴 killed")
            if st.button("Un-kill", key=f"unkill-{name}"):
                _toggle_kill(name, False)
                st.rerun()
        else:
            st.caption("🟢 live")
            if st.button("Kill", key=f"kill-{name}"):
                _toggle_kill(name, True)
                st.rerun()

with st.expander("Per-UID kill switches"):
    known_uids = sorted(set(signal_book.list_uids()) | set(run_log.list_uids()))

    if not known_uids:
        st.write("No UIDs recorded yet.")
    else:
        for uid in known_uids:
            uid_col, action_col = st.columns([5, 1])
            uid_killed = kill_switch.is_killed(uid)
            uid_col.write(("🔴 " if uid_killed else "🟢 ") + uid)
            with action_col:
                if uid_killed:
                    if st.button("Un-kill", key=f"unkill-uid-{uid}"):
                        _toggle_kill(uid, False)
                        st.rerun()
                else:
                    if st.button("Kill", key=f"kill-uid-{uid}"):
                        _toggle_kill(uid, True)
                        st.rerun()

st.divider()


# ============================================================
# Per-UID run health
# ============================================================

st.subheader("Run health")

latest_per_uid = run_log.get_latest_per_uid()

if latest_per_uid.empty:
    st.info(
        "No UIDs recorded yet. Run "
        "`live.execution.live_ems.run_live_day(..., run_log=...)` for a "
        "UID and refresh this page."
    )
else:
    now = pd.Timestamp.now()

    def _status(row: pd.Series) -> str:
        if kill_switch.is_killed(row["uid"]) or system_killed:
            return "⏸️ KILLED"
        if row["decision"] == "ERROR":
            return "🔴 ERROR"
        age_hours = (now - pd.Timestamp(row["ran_at"])).total_seconds() / 3600.0
        if age_hours > stale_after_hours:
            return "🟡 STALE"
        return "🟢 OK"

    health = latest_per_uid.copy()
    health["age_hours"] = (
        now - pd.to_datetime(health["ran_at"])
    ).dt.total_seconds() / 3600.0
    health["status"] = health.apply(_status, axis=1)

    display_columns = [
        "status",
        "uid",
        "decision",
        "as_of",
        "data_as_of",
        "age_hours",
        "ran_at",
    ]
    st.dataframe(
        health[display_columns].sort_values("status"),
        width="stretch",
        hide_index=True,
        column_config={
            "age_hours": st.column_config.NumberColumn(
                "hours since check", format="%.1f"
            ),
        },
    )

    ok_count = (health["status"] == "🟢 OK").sum()
    stale_count = (health["status"] == "🟡 STALE").sum()
    error_count = (health["status"] == "🔴 ERROR").sum()
    killed_count = (health["status"] == "⏸️ KILLED").sum()

    metric_cols = st.columns(4)
    metric_cols[0].metric("OK", int(ok_count))
    metric_cols[1].metric("Stale", int(stale_count))
    metric_cols[2].metric("Errors", int(error_count))
    metric_cols[3].metric("Killed", int(killed_count))
