"""Quick script to pull one day of bars for one symbol from futures_data_1h.duckdb."""
import duckdb

DB_PATH = "duckdb/futures_data.duckdb"
SYMBOL = "MES=F"
DAY = "2021-01-10"
HOUR_START = 20
HOUR_END = 22

con = duckdb.connect(DB_PATH, read_only=True)
df = con.execute(
    """
    SELECT *
    FROM bars
    WHERE symbol = ?
      AND CAST(timestamp AS DATE) = ?
      AND EXTRACT(HOUR FROM timestamp) BETWEEN ? AND ?
    ORDER BY timestamp
    """,
    [SYMBOL, DAY, HOUR_START, HOUR_END],
).fetchdf()

print(df)
