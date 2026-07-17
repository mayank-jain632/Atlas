# Atlas futures strategies

Single-symbol, one-contract futures framework. All math comes from the
top-level `indicators/` package (not a local copy) -- see
`strategies/futures/instruments.py` for the symbol registry.

Two data sources, two granularities:

- `duckdb/market_data.duckdb` (default, `timeframe="1d"`): daily bars for
  root contracts -- `ES=F`, `NQ=F`, `YM=F`, `RTY=F`, `GC=F`, `SI=F`, `CL=F`,
  `NG=F`, `ZN=F`, `ZB=F`.
- `duckdb/futures_data.duckdb` (`timeframe="1m"`): 1-minute bars for micro
  contracts that have their own dedicated series -- `MES=F`, `MNQ=F`,
  `MGC=F`, `MCL=F`, `MNG=F`, `MHG=F`, `SIL=F`. Pass both `db_path` and
  `timeframe="1m"` explicitly to `create_futures_strategy` to use this
  source; the strategies default to daily-bar semantics otherwise (ATR
  stop and close-to-close mark-to-market were designed around daily bars,
  so running on 1-minute data changes their character -- it isn't a
  drop-in swap of granularity).

MYM and M2K have no dedicated micro series yet, so they still price off
their root's daily bars (`YM=F`, `RTY=F`) regardless of which db is used.

Example UIDs:

```python
UID = "stema__s=MES__st_period=10__st_mult=3__ema=200__atr=14__sl_atr=2"
UID = "psarema__s=MNQ__psar_step=0.02__psar_max=0.20__ema=200__atr=14__sl_atr=2"
UID = "dcemachop__s=MGC__dc=20__ema=200__adx_period=14__adx=25__chop_period=14__chop=35__atr=14__sl_atr=2"
```

The framework currently assumes one contract, close-to-close mark-to-market, close fills, a fixed ATR stop from entry ATR, and no commissions, slippage, roll costs or margin enforcement.
