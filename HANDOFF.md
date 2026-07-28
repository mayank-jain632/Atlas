# Atlas — Handoff (server migration, 2026-07-28)

This is a status snapshot for whichever Claude session picks this up next, most
likely on the new server. It was written by the Claude session that built
almost everything described in "The live execution system" below. Read this
before touching `live/` or `dashboard/` — it explains what's real, what's
faked, and what the standing constraints are.

**The single most important fact in this document:** nothing in this
codebase has ever connected to a real broker. Every IBKR-related code path
has been built and tested against fakes only. If you're picking this up on
the server specifically to go live, treat that as the headline risk, not a
footnote — see "Going live for real" at the bottom before doing anything with
`USE_REAL_IBKR`.

---

## 1. What Atlas is (brief — this part is stable, not what changed recently)

Atlas is a Python backtesting framework for systematic trading strategies,
built on DuckDB for market data storage. Three-layer architecture:

```
Strategy → EMS (Execution Management System) → DataInterface
```

- `DataInterface`: DuckDB access, OHLCV, historical lookback, symbol/time iteration.
- `EMS` (`ems/ems.py`): portfolio state, cash, positions, `place_trade()`,
  `target_quantity()`, `rebalance_to_weights()`, tradebook, equity history.
- `Strategy`: pure trading logic, one method that matters today —
  `on_day_close()`. Strategy families live under `strategies/`: `momentum/`
  (price momentum, momentum + indicator filter, momentum diversity),
  `futures/` (STEMA/PSAREMA/DCEMACHOP-style trend strategies), `intervention/`.
- Strategies are identified by a `uid` string (e.g.
  `momentum__u=nasdaq100__sig=price__lb=90__rb=monthly__n=10__alloc=score`)
  that fully encodes its parameters — `build_uid()`/`parse_uid()` on every
  strategy class round-trip this.
- `runners/run_strats.py` is the standard single/multi-UID backtest runner;
  `runners/run_strats_grid.py` / `run_futures_grid.py` run parameter sweeps.
  `STRATEGY_CLASSES` and `MOMENTUM_STRATEGIES` in `run_strats.py` are the
  canonical UID-prefix → strategy-class mapping — the live scripts described
  below import these directly rather than duplicating the mapping.
- See `README.md` for the full original architecture writeup (predates
  everything in this document).

Full detail on this side of the system isn't repeated here — it hasn't
changed. Everything below is new.

---

## 2. The live execution system (this is the actual point of this doc)

### Design philosophy, stated once so it doesn't need repeating per file

- **Reuse backtest strategy classes unmodified.** Live wrapping happens via a
  mixin bolted on ahead of the class (`live_ems.py`), never by editing
  `strategies/`. `on_day_close()`, `rebalance_to_weights()`,
  `target_quantity()` run identically in live and backtest.
- **Build every integration against a fake first, connect for real last, and
  only when explicitly told to.** This applies to IBKR and to Telegram
  alerting equally. Nothing sends a real order or a real message anywhere in
  this codebase today.
- **A `typing.Protocol` decouples "what provides account state" from "how."**
  `FakeAccountStateProvider` and `IBKRClient` both satisfy
  `AccountStateProvider` structurally — no shared base class, no adapter.
  Swapping one for the other is a one-line change at the one call site that
  constructs it.
- **Every scheduled job logs unconditionally, success or failure,** because
  "nothing happened" and "it never ran" must be distinguishable from the
  outside (dashboard, alerts) without reading a log file on the box itself.
- **Kill switches are a dumb, shared, polled store — not an enforcement
  mechanism.** `KillSwitch` itself does nothing except answer "is this
  killed"; every script that should respect a switch checks it explicitly.
- One physical SQLite file (`live/execution/state/signal_book.sqlite`, path
  constant `DEFAULT_SIGNAL_BOOK_PATH`) backs `SignalBook`, `RunLog`, and
  `KillSwitch` — three tables, one file, WAL mode,
  `check_same_thread=False` (needed because Streamlit's `st.cache_resource`
  can hand a cached connection to a different thread across reruns).

### File-by-file reference: `live/execution/`

- **`signal_book.py`** — `SignalBook`. Append-only target-position ledger.
  `write_targets(uid, targets, as_of, reason, notes_by_symbol=None)`;
  `get_latest_targets(uid)` (dict, latest row per symbol);
  `get_latest_targets_detail(uid)` (full row incl. `notes` JSON — has
  `target_weight`/`score`/`rank` when a momentum strategy wrote it);
  `get_history(uid, symbol=None)`; `list_uids()`. A dropped symbol needs an
  explicit `target_quantity=0` row — there's no implicit "absent = zero."

- **`run_log.py`** — `RunLog`. Append-only "did this run" ledger, separate
  table, same file. `record(uid, as_of, decision, ran_at=None,
  data_as_of=None, detail=None)`. `decision` values in use today:
  `REBALANCED`, `NO_REBALANCE`, `ERROR`, `KILLED` (strategy jobs);
  `REFRESHED`, `NO_NEW_DATA`, `PARTIAL_FAILURE` (data-refresh job, plus
  `ERROR`/`KILLED` shared with the above). `get_latest_per_uid()` (System
  Health grid), `get_all_history()` (Alerts page), `get_history(uid)`,
  `get_latest(uid)`.

- **`kill_switch.py`** — `KillSwitch`. `DEFAULT_SWITCHES = (SYSTEM_SWITCH,
  "market_data", "strategy_engine", "oms_trader", "alerting")`, pre-seeded on
  construction; per-UID switches are created on first use. `is_killed(name)`
  is true if `name` is killed *or* `"system"` is killed — two-level
  hierarchy, baked in once. `set_killed(name, killed, reason="",
  updated_by="dashboard")` upserts. Adding a new name to `DEFAULT_SWITCHES`
  is enough to get it a kill/unkill button on the System Health dashboard
  page for free — that page's process-switch UI is fully generic over
  whatever's in the tuple.

- **`account_state.py`** — `AccountState` (frozen dataclass: `cash`,
  `positions: dict[str, float]`, `equity`, `as_of`), `AccountStateProvider`
  (`Protocol`, one method `get_account_state()`),
  `FakeAccountStateProvider` (frozen dataclass wrapping one fixed
  `AccountState`, always returns it). This is the whole seam that lets
  everything downstream not care whether it's talking to a fake or IBKR.

- **`live_ems.py`** — the live-wrapping machinery.
  `make_live_strategy(strategy_class, *, signal_book, **kwargs)` builds
  `type("Live"+name, (SignalBookLiveMixin, strategy_class), {})` and
  constructs it. `SignalBookLiveMixin`: overrides `place_trade()` to call
  `super().place_trade()` (unmodified simulated bookkeeping still happens —
  `rebalance_to_weights()` reads `self.cash`/`self.positions` off it) and
  buffers the result; `flush_signal_book()` writes the buffer as one batch
  and returns whether anything was written; `seed_account_state(state)`
  overwrites `self.cash`/`self.positions` with real numbers before the
  strategy computes anything (does *not* touch `tradebook`/`equity_history`
  — those only need to cover today's single call). Module function
  `run_live_day(strategy, as_of, account_state_provider=None, run_log=None,
  data_as_of=None)` is the actual sequence: seed (if a provider given) →
  `set_current_timestamp()` → `on_day_close()` → `flush_signal_book()` →
  record to run_log (`REBALANCED`/`NO_REBALANCE` on success, `ERROR` +
  re-raise on failure).

- **`ibkr_client.py`** — `IBKRClient(host="127.0.0.1", port=7497,
  client_id=1, readonly=True, reconnect_attempts=3,
  reconnect_delay_seconds=5.0)`. Wraps `ib_async`'s `IB()`; `ib_async` is
  imported lazily inside `connect()`/`get_last_prices()`, never at module
  level, so this file is importable/testable with zero real dependency on
  the package being usable. `connect()`/`disconnect()`/`is_connected()` own
  the lifecycle. `ensure_connected()` is what a scheduler should call
  instead of `connect()` directly — no-ops if already connected, otherwise
  retries with a delay between attempts, raises `ConnectionError` after
  exhausting attempts. `get_account_state(account=None)` satisfies
  `AccountStateProvider` (translates `accountSummary()` + `positions()`,
  filtered to the resolved account). `get_last_price(s)`/`get_last_prices()`
  do a one-shot `reqTickers()` snapshot (not a streaming subscription —
  polling was decided to be sufficient for daily momentum), filtering
  ib_async's NaN "no trade" sentinel. **Port note:** 7497 default is TWS
  paper; IB Gateway paper is 4002, TWS live 7496, IB Gateway live 4001 —
  confirm against whatever's actually running on the server before
  connecting anything.

- **`oms.py`** — reconciliation logic (no order placement). `Order` (frozen
  dataclass: `symbol`, signed `quantity` — positive=buy, negative=sell,
  matching `EMS.place_trade()`'s existing convention exactly; `.action`
  property derives `"BUY"`/`"SELL"` from the sign). `compute_orders(targets,
  actual_positions, min_quantity=1e-6)` diffs the two dicts per symbol
  (`delta = target - actual`); a symbol held but never targeted at all
  (not even an explicit zero row) is treated as an implicit zero-target, so
  stray/manual positions still get closed; deltas below `min_quantity` are
  dropped (fractional-share rounding noise). `reconcile_uid(uid,
  signal_book, account_state_provider)` composes
  `signal_book.get_latest_targets()` + `account_state_provider
  .get_account_state().positions` into that call. **This is as far as OMS
  goes today — nothing places these orders.**

- **`alerting.py`** — `TelegramNotifier(bot_token, chat_id,
  kill_switch=None, timeout_seconds=10.0)`. Chosen over Discord because
  another of the user's programs already uses Telegram. `requests` is
  imported lazily inside `_post()`. `send(text)` checks
  `kill_switch.is_killed("alerting")` first — muted means log-and-return-
  `False`, never raise; a real network/API failure during an actual send
  *does* raise (silent alerting failure is worse than a loud one).
  `notify_error(uid, error)`, `notify_kill_switch(name, killed, reason="")`
  are the two message shapes in use. `notifier_from_env(kill_switch=None)`
  builds one from `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` env vars, or
  returns `None` if either is unset — this is the only place "not
  configured" is handled; every caller just gets `None` back and checks it.
  **Neither env var is set anywhere today** — alerting is fully wired but
  currently a no-op end to end.

### The scheduled scripts: `live/`

- **`run_daily_check.py`** — the strategy-signal cron script (not yet
  actually scheduled — no crontab/systemd entry exists). Config at the top:
  `UIDS` (list, currently one nasdaq100 momentum UID), `CAPITAL`, `DB_PATH`,
  `TIMEFRAME`, `UNIVERSE_ROOT`, `STATE_DB_PATH`, `ACCOUNT_STATE_PROVIDER`
  (default `None`), `USE_REAL_IBKR` (default `False`, plus
  `IBKR_HOST`/`IBKR_PORT`/`IBKR_CLIENT_ID`), `LOG_PATH`
  (`live/logs/daily_check.log`). `main()`: for each UID, checks
  `kill_switch.is_killed("strategy_engine")` and `is_killed(uid)` (skip +
  log `KILLED` if either), builds a live strategy via
  `build_live_strategy()` (resolves the strategy class from the UID prefix
  using `runners.run_strats.STRATEGY_CLASSES`/`MOMENTUM_STRATEGIES` — not a
  separate mapping), calls `run_live_day()`. Any exception at any stage is
  caught per-UID, logged to run_log as `ERROR`, pushed through
  `notifier_from_env()`'s notifier if one exists, and does **not** stop
  other UIDs in the list. If `USE_REAL_IBKR` is `True`, an `IBKRClient` is
  constructed, `ensure_connected()`-ed, used as the account-state provider
  for the whole run, and `disconnect()`-ed in a `finally` no matter what
  happens. Exit code: 0 if every UID succeeded, 1 otherwise.

- **`run_data_refresh.py`** — the market-data cron script (also not
  scheduled yet), same shape. Checks the `market_data` kill switch, loads
  every symbol across the configured universes (`UniverseManager`, a plain
  local CSV read — no network), runs an **incremental** download
  (`incremental=True`, only new bars per symbol since the last stored one)
  through the existing `data/yahoo_downloader.py::YahooDownloader`. Logs to
  the *same* run_log table under the synthetic UID
  `"market_data_refresh"` — deliberately reusing System Health/Alerts
  rather than building separate observability. Decision values:
  `REFRESHED`/`NO_NEW_DATA`/`PARTIAL_FAILURE`/`ERROR`/`KILLED`. Notifies on
  `ERROR` and `PARTIAL_FAILURE`. Exit 0/1 same convention.

- **`check_ibkr_connection.py`** — not part of either cron job, never
  invoked automatically. A manual smoke-test script for a human to run
  later: connect → print account state → print one test price →
  disconnect. `run_check(client, symbol)` is factored out so its sequencing
  is unit-tested against a fake client; the script itself has never been
  run against a real gateway.

### The dashboard: `dashboard/`

Streamlit multi-page app (`st.navigation`), entrypoint `dashboard/app.py`.
Shared helpers in `dashboard/lib.py`: cached (`st.cache_resource`)
`SignalBook`/`RunLog`/`KillSwitch` instances keyed off a sidebar-editable
`db_path` (session-state key `db_path`, defaults to
`DEFAULT_SIGNAL_BOOK_PATH`). Pages, and how real each one is:

- **`system_health.py`** — real. Kill switch grid (system-wide, all of
  `DEFAULT_SWITCHES`, per-UID) with kill/unkill buttons that call
  `_toggle_kill()`, which flips the switch *and* calls
  `notifier.notify_kill_switch()` if a notifier is configured. Per-UID run
  health grid from `run_log.get_latest_per_uid()` (🟢 OK / 🟡 STALE / 🔴
  ERROR / ⏸️ KILLED, staleness threshold configurable in the sidebar).
- **`signals.py`** — real. One UID at a time: last decision + staleness,
  current target portfolio (charts `target_weight` from the signal book's
  `notes` JSON when present, falls back to raw share count with an
  explanatory caption otherwise), full run history, full signal-book
  history in an expander.
- **`alerts.py`** — real. Cross-UID filterable run_log view — decision tabs
  (dynamically derived from whatever decision values actually exist, so new
  ones like `REFRESHED` need no page changes), UID filter, text search on
  `detail`.
- **`data.py`** — real. DuckDB coverage overview across every `.duckdb`
  file under `duckdb/`, plus a per-symbol gap check (default `gap_multiple
  =4.5`, tuned against real gaps like 9/11 and Hurricane Sandy vs. ~230
  false positives at a lower threshold). Opens/closes a fresh read-only
  connection per query rather than holding one open, to avoid lock
  contention with cron-run writers.
- **`oms.py`** — real reconciliation logic, hypothetical input. Shows real
  signal-book targets, but "actual positions" is an editable table
  defaulting to zero (there's no real broker to read positions from yet) —
  explicitly labeled as such. Runs `compute_orders()` against whatever's in
  the editor and shows the resulting BUY/SELL orders. Swapping the editable
  table for `account_state_provider.get_account_state().positions` is the
  only change needed once IBKR is connected.
- **`live_data.py`**, **`live_vs_backtest.py`** — honest placeholders,
  explaining specifically what's missing rather than showing fake data.

### Testing conventions (established across every file above — keep following these)

- Nothing that touches a real broker, a real Telegram send, or the real
  network is ever exercised for real in a test. IBKR: hand-built fakes
  (`_FakeIB`, `_FakeAccountValue`, `_FakePosition`, `_FakeTicker`) injected
  directly as `client._ib`, bypassing `connect()` entirely.
  `ensure_connected()`'s retry logic is tested by monkeypatching
  `client.connect` itself. Telegram: `notifier._post` is monkeypatched;
  every test file that touches `run_daily_check`/`run_data_refresh`
  explicitly `monkeypatch.delenv("TELEGRAM_BOT_TOKEN"/"TELEGRAM_CHAT_ID")`
  so a real env var on the test machine can never leak through.
  Market-data refresh: `YahooDownloader` itself is replaced with a fake
  class matching its constructor + context-manager protocol.
- Module-level config constants (`UIDS`, `STATE_DB_PATH`, `LOG_PATH`,
  `USE_REAL_IBKR`, etc.) are overridden per-test via
  `monkeypatch.setattr(module, "NAME", value)` — never edit the script
  files themselves to test them.
- Dashboard pages are tested headlessly via `streamlit.testing.v1.AppTest`
  — no browser, no Playwright. Known gap: this Streamlit version's
  `AppTest` has no interaction support for `st.data_editor`, so the OMS
  page's editable-table interactions aren't UI-tested, only its default
  state; the reconciliation logic itself is fully covered independently in
  `tests/test_oms.py`.
- Full suite: `pytest tests/ -q` from the repo root with the venv active
  (`source venv/bin/activate` — **use the project's venv, not system
  Python**; a past bug on this machine was `python3`/`streamlit` resolving
  to `/Library/Frameworks/Python.framework/...` instead of `venv/`, which
  silently breaks `duckdb.connect` by treating the local `duckdb/` data
  directory as a namespace package instead of importing the real library).
  220 tests passing as of this handoff.

---

## 3. What's real vs. what's still fake — the one-glance version

| Piece | Status |
|---|---|
| Backtest engine, strategies, DuckDB storage | Real, long-standing |
| Signal book / run log / kill switch (SQLite) | Real, real data flows through these on every local test run |
| Live strategy wrapping (`live_ems.py`) | Real logic, runs real unmodified strategies |
| Account-state seeding | Real mechanism; source is `FakeAccountStateProvider` (fixed fake) in every run today |
| IBKR connection | Built, unit-tested against fakes, **never connected for real** |
| OMS reconciliation | Real logic; "actual positions" input is a fake/editable stand-in |
| Telegram alerting | Built, unit-tested against fakes; **no env vars set anywhere**, so it's a no-op in practice |
| `run_daily_check.py` / `run_data_refresh.py` | Real scripts, run and tested locally by hand; **no cron/systemd entry exists** |
| Dashboard | Real for System Health / Signals / Alerts / Data / OMS; honest placeholders for Live Data / Live vs Backtest |

---

## 4. Next steps — this is what the server migration is actually for

Roughly in dependency order:

1. **Get the environment right first.** Recreate the venv on the server,
   `pip install -r requirements.txt`. Confirm `python3`/`streamlit`/`pytest`
   all resolve inside that venv, not system Python (see the testing-section
   note above — this exact bug happened twice already). Run `pytest tests/
   -q` and confirm all pass before changing anything.

2. **Wire the actual scheduling mechanism.** Neither
   `run_daily_check.py` nor `run_data_refresh.py` has a crontab/systemd
   timer yet — that was explicitly deferred to "once the server exists,"
   which is now. Sequence `run_data_refresh.py` before `run_daily_check.py`
   each day (fresh data before the strategy decides anything). The
   reference production system this design borrowed from registers each
   job **twice**, at two different UTC times, with a DST-aware wrapper that
   lets only the correct invocation actually execute — worth adopting once
   there's a real timezone-crossing schedule to protect, not built yet.

3. **Set up IB Gateway/TWS + IBC for headless login**, if going live is
   actually the next goal (not just "the server is ready"). This is
   genuinely new work, not just flipping a flag: the gateway process itself
   has to run continuously, survive restarts, and get through 2FA somehow
   without a human present — that's an IBC (IBController) configuration
   question, not something this codebase currently does anything about.

4. **Confirm the port** (`IBKR_PORT` in `run_daily_check.py`) matches
   whatever's actually configured — 7497 (TWS paper) is the current
   default; IB Gateway paper is 4002. Getting this wrong is a connection
   failure, not a wrong-account trade, so low risk to get wrong once, but
   worth just checking first.

5. **Run `live/check_ibkr_connection.py` by hand first**, once Gateway is
   up, before touching `USE_REAL_IBKR` in the scheduler. It's a manual,
   read-only smoke test (connect, read account state, read one price,
   disconnect) that's never been run against a real gateway — this is the
   first time any of this code will see a real IBKR connection at all.
   Expect to debug something here; nothing about the real Gateway's exact
   behavior has been verified yet, only the translation logic against
   fakes.

6. **Only after that succeeds**, flip `USE_REAL_IBKR = True` in
   `run_daily_check.py`, with `readonly=True` kept on initially (it's the
   default) so even a fully-wired real connection still can't place
   orders. Run one real day's `run_daily_check.py` manually (not via cron
   yet) and check the dashboard: does `AccountState.equity` roughly match
   what `get_positions()`/`portfolio_value()` compute from local prices?
   (See `account_state.py`'s docstring — a large gap means stale/wrong
   local price data, not a bug in seeding itself.)

7. **Set `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`** wherever the scheduler
   and dashboard actually run, once you have them, then trigger one
   deliberate failure (or flip a kill switch from the dashboard) to confirm
   a real Telegram message actually arrives before trusting it silently.

8. **OMS / real order placement does not exist yet.** `oms.py` computes
   what orders *would* be needed; nothing places them. If placing real
   orders is the next milestone after this handoff, that's new design work,
   not a flag to flip — decide execution style (market/limit, how to handle
   partial fills, fractional-share support on the actual broker) before
   writing it.

9. **Standing constraint, unprompted every time it came up in this session
   and worth repeating here explicitly: never place a real order, and
   never flip anything toward "live" (real money, non-readonly) without
   the user explicitly asking for that specific step.** Building and
   testing against fakes is always fine to do proactively; connecting for
   real is not.

---

## 5. Quick file map

```
live/
├── run_daily_check.py      # strategy-signal cron script (not scheduled yet)
├── run_data_refresh.py     # market-data cron script (not scheduled yet)
├── check_ibkr_connection.py# manual, human-run-only IBKR smoke test
├── logs/                   # daily_check.log, data_refresh.log (gitignored)
├── dashboard/               # (empty placeholder dir, unrelated to dashboard/ at repo root)
└── execution/
    ├── signal_book.py       # target-position ledger (SQLite)
    ├── run_log.py           # "did this run" ledger (SQLite, same file)
    ├── kill_switch.py       # named on/off flags (SQLite, same file)
    ├── account_state.py     # AccountState / AccountStateProvider Protocol / Fake
    ├── live_ems.py           # SignalBookLiveMixin, make_live_strategy, run_live_day
    ├── ibkr_client.py        # IBKRClient (never connected for real)
    ├── oms.py                # Order, compute_orders, reconcile_uid
    ├── alerting.py           # TelegramNotifier, notifier_from_env
    └── state/                # signal_book.sqlite (gitignored, **/execution/state/)

dashboard/
├── app.py                   # st.navigation entrypoint
├── lib.py                   # shared cached SignalBook/RunLog/KillSwitch + db_path handling
└── pages/
    ├── system_health.py      # real
    ├── signals.py             # real
    ├── alerts.py               # real
    ├── data.py                  # real
    ├── oms.py                    # real logic, hypothetical input
    ├── live_data.py                # placeholder
    └── live_vs_backtest.py          # placeholder

tests/
    test_signal_book.py, test_run_log.py, test_kill_switch.py,
    test_account_state.py, test_live_ems.py, test_ibkr_client.py,
    test_oms.py, test_oms_page.py, test_alerting.py,
    test_run_daily_check.py, test_run_data_refresh.py,
    test_check_ibkr_connection.py, test_dashboard.py
```

---

## 6. Conventions worth preserving if you keep extending this

- New scheduled jobs should log to the *same* `run_log`/`signal_book`/
  `kill_switch` file under a synthetic UID if they're not strategy-shaped
  (see `run_data_refresh.py`'s `"market_data_refresh"`) — the dashboard
  picks up new UIDs automatically, no page changes needed.
- New kill switches: add the name to `DEFAULT_SWITCHES` in
  `kill_switch.py` and it gets dashboard UI for free.
- New `AccountStateProvider` implementations just need one method,
  `get_account_state() -> AccountState` — no inheritance required.
- Every "is this configured" check (IBKR, Telegram) degrades to `None` /
  a safe default rather than raising, so the rest of the pipeline runs
  identically whether or not that integration exists yet.
- Real IBKR/Telegram config lives in environment variables, never in
  source — `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` today; expect IBKR
  credentials (if any beyond host/port/client_id) to follow the same
  pattern.
