# Data Layer

## Contract
Input: `query(asset_code: str, start: date, end: date)`
Output: `pd.DataFrame` with columns `[date, open, high, low, close, volume]`, sorted by date ascending (empty DataFrame with those columns if asset not locally present). Prices returned by `query()` / `read_local()` are locally reconstructed HFQ prices.

## Implementation Notes
- `store.py` — Parquet read/write, one file per asset under `data/db/{asset_code}.parquet`
- Stored Parquet schema is raw bars plus factor: `[date, raw_open, raw_high, raw_low, raw_close, volume, adj_factor]`
- `store.read_storage()` returns the raw stored schema; `store.read_local()` and `store.query()` project raw bars to HFQ using the first stored `adj_factor` as the fixed baseline: `raw_price * adj_factor / first_adj_factor`
- `sync.py` — Tushare incremental sync. Fetches raw ETF bars via `ts.pro_bar(asset="FD", adj=None)` and fund adjustment factors via `pro.fund_adj()`, then left-merges factors onto raw trading days
- `config.py` — loads `TUSHARE_TOKEN` from env or `.env` (python-dotenv)
- Incremental cursor: `start_date = max(local_date) + 1 day`; first-time sync starts from `_HISTORY_START = "20130101"`
- Column rename on write: `trade_date → date`, `vol → volume` (see `_COLUMN_MAP`)
- Adjustment-factor fill policy: missing raw trading-day factors are forward-filled; only a leading missing prefix may use the first future factor
- Dedup policy: `drop_duplicates(subset=["date"], keep="first")` — existing history wins; incremental sync must not rewrite prior rows because the local HFQ baseline is fixed
- Rate-limit handling: retries up to 3 times, sleeps 60s between attempts when error message contains `rate`/`40203`/`freq`/`exceed`
- CLI entry: `python -m data` (see `__main__.py`)

## Pitfalls
- `query()` returns an **empty** DataFrame (not an error) when the asset has no local file — callers must check `len(df) > 0` before indexing
- `date` column is a `datetime64[ns]`, not a `date` object. Comparing with `datetime.date` works via `pd.Timestamp` coercion but mixing raw `date` in a mask will silently mis-compare
- Existing legacy qfq Parquet files must be rebuilt before incremental raw+factor sync can append to them; `merge_and_save()` raises on legacy schema to avoid mixed qfq/raw history
- `sync()` returns the raw tushare row count (including duplicates with existing data); it is **not** the net-new row count after dedup
- If `TUSHARE_TOKEN` is missing, `sync()` raises at call time; `query()` works offline with no token
