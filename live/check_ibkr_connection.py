"""
Manual, one-off connectivity check for IBKR.

NOT part of the daily scheduled pipeline (live/run_daily_check.py) and
never invoked automatically by anything in this codebase -- this
genuinely connects to a broker, so it only runs when a human runs it
on purpose.

Meant to be run by hand, later, on the server: once IB Gateway/TWS is
actually installed and running, reachable at the given host/port, and
credentials/2FA are sorted out (see the module docstring in
live/execution/ibkr_client.py for what "actually connected" requires).
Proves the full round trip -- connect, read account state, read one
price, disconnect -- works against a real paper account before
flipping USE_REAL_IBKR on in run_daily_check.py.

Usage:
    python3 live/check_ibkr_connection.py \
        --host 127.0.0.1 --port 7497 --client-id 1 --symbol AAPL
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live.execution.ibkr_client import IBKRClient


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=7497,
        help="TWS paper=7497, TWS live=7496, IB Gateway paper=4002, IB Gateway live=4001",
    )
    parser.add_argument("--client-id", type=int, default=1)
    parser.add_argument(
        "--symbol", default="AAPL", help="Symbol to test a price lookup against"
    )
    return parser.parse_args(argv)


def run_check(client: IBKRClient, symbol: str) -> None:
    """
    The actual sequence of calls this check makes, factored out from
    main() so it's exercisable against a fake client in tests -- proving
    the sequencing and error handling are correct without a real
    connection, same discipline as ibkr_client.py's own tests.
    """
    print(f"Connecting to {client.host}:{client.port} (client id {client.client_id})...")
    client.ensure_connected()
    print("Connected.")

    print("\nAccount state:")
    account_state = client.get_account_state()
    print(f"  cash:      {account_state.cash:,.2f}")
    print(f"  equity:    {account_state.equity:,.2f}")
    print(f"  positions: {account_state.positions or '(none)'}")

    print(f"\nPrice lookup ({symbol}):")
    price = client.get_last_price(symbol)
    print(f"  last: {price}")

    print("\nDisconnecting...")
    client.disconnect()
    print("Done -- connection round-trip succeeded.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    client = IBKRClient(
        host=args.host, port=args.port, client_id=args.client_id, readonly=True
    )

    try:
        run_check(client, args.symbol)
    except Exception as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        client.disconnect()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
