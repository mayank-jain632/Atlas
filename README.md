# Atlas

Atlas is a lightweight event-driven quantitative trading framework for developing, backtesting, and eventually deploying systematic trading strategies across multiple asset classes.

The framework is intentionally simple. Every strategy follows the same architecture, making it easy to implement new ideas while reusing the same data access, portfolio management, accounting, and tradebook infrastructure.

The initial focus is on:

- Stock momentum strategies
- ETF intervention strategies
- Futures trend-following strategies

The framework is designed so that live trading can later be added with minimal changes.

---

# Architecture

Atlas consists of three layers.

```
Strategy
    ↑
EMS
    ↑
DataInterface
```

## DataInterface

The `DataInterface` provides all market data required by strategies.

Responsibilities:

- Connect to DuckDB
- Access historical market data
- Provide current OHLCV values
- Historical lookback functions
- Symbol discovery
- Time iteration

Typical functions include:

```python
get_open(symbol)
get_high(symbol)
get_low(symbol)
get_close(symbol)

history(symbol, field, bars)

symbols()
timestamps()
```

The DataInterface is completely independent of any strategy.

---

## EMS

The EMS (Execution Management System) extends the DataInterface and provides the common trading infrastructure.

Responsibilities include:

- Event loop
- Portfolio state
- Cash management
- Position management
- Trade generation
- Standardized tradebook
- Equity tracking

Strategies never write tradebooks directly.

Instead they simply call:

```python
place_trade(...)

target_quantity(...)

target_percent(...)

close_position(...)

rebalance_to_weights(...)
```

The EMS automatically updates positions and records trades.

---

## Strategy

Every strategy inherits from the EMS.

A strategy only contains trading logic.

Currently the primary callback is

```python
on_day_close()
```

Later versions may include

```python
on_hour_close()

on_minute_close()

on_tick()
```

without changing existing strategies.

---

# Project Structure

```
atlas/

│
├── config/
│
│   ├── universes/
│   │
│   │   ├── sp500.csv
│   │   ├── nasdaq100.csv
│   │   ├── dow30.csv
│   │   ├── etfs.csv
│   │   └── futures.csv
│
├── duckdb/
│
│   └── market_data.duckdb
│
├── data/
│
│   ├── interface.py
│   ├── universe.py
│   └── yahoo_downloader.py
│
├── ems/
│
│   ├── ems.py
│   └── __init__.py
│
├── strategies/
│
│   ├── base.py
│   │
│   ├── momentum/
│   │
│   ├── intervention/
│   │
│   └── futures/
│
├── indicators/
│
├── analytics/
│
├── optimization/
│
└── runners/
```

---

# Market Data

Atlas stores all market data inside DuckDB.

```
duckdb/

    market_data.duckdb
```

The expected table is

```
bars(

    timestamp,

    symbol,

    timeframe,

    open,

    high,

    low,

    close,

    volume
)
```

Historical data can also be stored as Parquet files.

The DataInterface provides a common abstraction over both.

---

# Universes

Universes are stored as CSV files.

Examples include

```
sp500.csv

nasdaq100.csv

dow30.csv

etfs.csv

futures.csv
```

Strategies simply request a universe by name.

For example

```python
UniverseManager().load("sp500")
```

No strategy should hardcode symbol lists.

---

# Tradebook

All strategies produce the same tradebook format.

Each row represents a single executed trade.

Fields include

```
timestamp

uid

strategy

symbol

action

quantity

price

value

position_before

position_after

cash_before

cash_after

reason

notes
```

The `notes` field is strategy-specific metadata.

Examples:

Momentum:

```json
{
    "score":0.18,
    "rank":2
}
```

Futures:

```json
{
    "atr":22.5,
    "supertrend":1
}
```

Intervention:

```json
{
    "intervention":"drawdown_exit"
}
```

---

# Strategies

Strategies are organized by family.

```
strategies/

    momentum/

    intervention/

    futures/
```

Each family can contain multiple implementations.

For example

```
momentum/

    momentum.py

    momentum_diversity.py
```

or

```
futures/

    supertrend_ma_atr.py

    donchian_adx.py

    psar_ma_atr.py
```

Every strategy should expose

```python
build_uid()

parse_uid()

default_parameters()
```

allowing the framework to reconstruct strategy parameters directly from the UID.

---

# Philosophy

Strategies should answer only one question:

> Given the current market state, what should the portfolio own?

Everything else—including data access, portfolio accounting, tradebooks, and execution—is handled by Atlas.

This separation keeps strategies concise, reusable, and easy to test while allowing all strategies to share a common infrastructure.

---

# Current Development Roadmap

The initial implementation will focus on:

1. Momentum Strategies
   - Price Momentum
   - RSI Momentum
   - Volatility Adjusted Momentum
   - Momentum Diversity

2. ETF Intervention Strategies
   - Moving Average
   - Supertrend
   - MACD
   - RSI
   - Price Action

3. Futures Trend Strategies
   - Supertrend + MA + ATR
   - Donchian + ADX + Chop
   - PSAR + MA + ATR
   - Additional composite strategies

All strategy families will share the same DataInterface, EMS, tradebook, and portfolio management framework.