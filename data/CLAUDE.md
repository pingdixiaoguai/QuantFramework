# Data Layer

## Contract
Input: `query(asset_code: str, start: date, end: date)`
Output: `pd.DataFrame` with columns `[date, open, high, low, close, volume]`, sorted by date ascending (empty DataFrame with those columns if asset not locally present)

## Implementation Notes
- `store.py` — Parquet read/write, one file per asset under `data/db/{asset_code}.parquet`
- `sync.py` — Tushare incremental sync. Fetches via `ts.pro_bar(asset="FD", adj="qfq")` (fund/ETF, forward-adjusted prices)
- `config.py` — loads `TUSHARE_TOKEN` from env or `.env` (python-dotenv)
- Incremental cursor: `start_date = max(local_date) + 1 day`; first-time sync starts from `_HISTORY_START = "20160101"`
- Column rename on write: `trade_date → date`, `vol → volume` (see `_COLUMN_MAP`)
- Dedup policy: `drop_duplicates(subset=["date"], keep="last")` — newer records overwrite older ones
- Rate-limit handling: retries up to 3 times, sleeps 60s between attempts when error message contains `rate`/`40203`/`freq`/`exceed`
- CLI entry: `python -m data` (see `__main__.py`)

## Pitfalls
- `query()` returns an **empty** DataFrame (not an error) when the asset has no local file — callers must check `len(df) > 0` before indexing
- `date` column is a `datetime64[ns]`, not a `date` object. Comparing with `datetime.date` works via `pd.Timestamp` coercion but mixing raw `date` in a mask will silently mis-compare
- Sync writes `qfq`-adjusted prices only. Raw unadjusted prices are not stored — switching adjustment mode means re-syncing the whole history
- `sync()` returns the raw tushare row count (including duplicates with existing data); it is **not** the net-new row count after dedup
- If `TUSHARE_TOKEN` is missing, `sync()` raises at call time; `query()` works offline with no token
