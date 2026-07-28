"""
Atlas -- OMS page.

Shows what reconciliation would order right now to bring a broker book
in line with the signal book's current targets -- see
live/execution/oms.py's compute_orders()/reconcile_uid(), built and
tested against fakes exactly like the rest of the live pipeline.

No real IBKR connection exists yet (see live/execution/ibkr_client.py),
so there's no real "actual positions" to reconcile against. Rather
than fake that away, this page is explicit about it: it defaults to
assuming zero actual positions and lets you edit that assumption to
explore what reconciliation would produce for a hypothetical starting
book. Once IBKR is connected, "assumed actual positions" becomes a
real account_state_provider.get_account_state().positions call instead
-- the reconciliation logic underneath doesn't change either way.
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
from live.execution.oms import compute_orders

db_path = lib.init_db_path()
signal_book = lib.get_signal_book(db_path)

st.title("OMS")
st.info(
    "No order-management process runs yet, and there's no real broker "
    "connection to reconcile against. This shows what reconciliation "
    "*would* produce, using the real reconciliation logic, against an "
    "assumed starting position you edit below -- not a real one."
)

known_uids = signal_book.list_uids()

if not known_uids:
    st.info(
        "No UIDs recorded yet at this DB path. Run "
        "`live.execution.live_ems.run_live_day(..., run_log=...)` for a "
        "UID and refresh this page."
    )
    st.stop()

selected_uid = st.sidebar.selectbox("UID", known_uids)

if st.sidebar.button("Refresh"):
    st.rerun()


# ============================================================
# Target positions (real -- from the signal book)
# ============================================================

st.subheader("Target positions")
st.caption("From the signal book -- the same targets the Signals page shows.")

raw_targets = signal_book.get_latest_targets(selected_uid)
targets = {symbol: qty for symbol, qty in raw_targets.items() if abs(qty) > 1e-9}

if not targets:
    st.write("No open targets for this UID.")
else:
    st.dataframe(
        pd.DataFrame(
            [
                {"symbol": symbol, "target_quantity": qty}
                for symbol, qty in sorted(targets.items())
            ]
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "target_quantity": st.column_config.NumberColumn(
                "target quantity", format="%.4f"
            ),
        },
    )


# ============================================================
# Assumed actual positions (hypothetical -- editable)
# ============================================================

st.subheader("Assumed actual positions")
st.caption(
    "Defaults to zero for every targeted symbol -- edit to explore "
    "reconciliation against a hypothetical starting book. Add rows for "
    "symbols the account might hold that aren't in the current targets."
)

default_actual = pd.DataFrame(
    [{"symbol": symbol, "actual_quantity": 0.0} for symbol in sorted(targets)]
)

edited_actual = st.data_editor(
    default_actual,
    width="stretch",
    hide_index=True,
    num_rows="dynamic",
    key=f"oms_actual_positions_{selected_uid}",
    column_config={
        "actual_quantity": st.column_config.NumberColumn(
            "actual quantity", format="%.4f"
        ),
    },
)

actual_positions = {
    str(row["symbol"]).strip(): float(row["actual_quantity"])
    for _, row in edited_actual.iterrows()
    if str(row["symbol"]).strip()
}


# ============================================================
# Reconciliation
# ============================================================

st.subheader("Orders reconciliation would generate")

orders = compute_orders(targets, actual_positions)

if not orders:
    st.success("Assumed actual positions already match targets -- no orders needed.")
else:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "symbol": order.symbol,
                    "action": order.action,
                    "quantity": abs(order.quantity),
                }
                for order in orders
            ]
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "quantity": st.column_config.NumberColumn("quantity", format="%.4f"),
        },
    )
