# Strict Defensive ETF Pool — Research Memo

## Scope rule

Only the following three sleeves are eligible:

1. Dividend or dividend-low-volatility equity ETFs whose tracked index has an
   explicit dividend-history and/or low-volatility selection rule.
2. Sovereign-bond ETFs diversified by duration, plus one short-financing
   credit-bond ETF as a separately controlled credit sleeve.
3. Exchange-traded money-market funds as the cash sleeve.

Sector, broad-market, growth, technology, overseas broad-equity, commodity,
gold, and convertible-bond ETFs are out of scope. An ETF enters
the backtest only on its real first local trading date; no index proxy may be
used to extend its history before listing.

## Recommended v1 candidate pool

| Sleeve | Code | Fund | Earliest usable date | Selection reason |
|---|---|---|---|---|
| Dividend equity | 510880.SH | 华泰柏瑞上证红利ETF | 2007-01-18 | Tracks the SSE Dividend Index. The index requires sustained cash dividends and ranks on dividend yield. |
| Dividend-low-vol equity | 512890.SH | 华泰柏瑞中证红利低波动ETF | 2019-01-18 | Explicit high-dividend and low-volatility equity exposure. |
| Dividend-low-vol equity | 515450.SH | 南方标普中国A股大盘红利低波50ETF | 2020-02-26 | Explicit large-cap dividend-low-volatility exposure. |
| Sovereign bonds | 511010.SH | 国泰上证5年期国债ETF | 2013-03-25 | Five-year Treasury duration sleeve. |
| Sovereign bonds | 511260.SH | 国泰上证10年期国债ETF | 2017-08-24 | Ten-year Treasury duration sleeve. |
| Sovereign bonds | 511090.SH | 鹏扬中债-30年期国债ETF | 2023-06-13 | Long-duration Treasury sleeve. |
| Credit short bond | 511360.SH | 海富通中证短融ETF | 2020-09-25 | Short-financing credit sleeve; controlled separately from sovereign duration risk. |
| Cash | 511880.SH | 银华日利ETF | 2013-04-18 | Exchange-traded money-market cash sleeve with an HFQ price series suitable for the engine. |

## Deliberate exclusions

- 159915.SZ 创业板 ETF, 510300.SH 沪深300 ETF, 510500.SH 中证500 ETF:
  broad/growth equity, not dividend or low-volatility mandates.
- 513050.SH, 513100.SH, 513180.SH, 513500.SH, 159941.SZ:
  overseas internet, technology, or broad-equity exposure, not a defensive
  equity mandate.
- 512800.SH 银行 ETF:
  a single-sector concentration, not a diversified dividend/low-volatility
  index mandate.
- 518880.SH 黄金 ETF:
  explicitly excluded by the strategy scope.
- Other corporate-credit and all convertible-bond ETFs:
  excluded; 511360.SH is the sole pre-specified credit-bond exception.

## Confirmed v2.1 strategy

The strategy is a monthly winner-take-all rotation across the eight confirmed
defensive ETFs. On the first A-share trading day of a month, after the close,
it calculates each available ETF's 20-trading-day momentum and Kaufman
efficiency ratio (ER):

`score = (C_t / C_(t-20) - 1) × |C_t - C_(t-20)| / Σ|C_i - C_(i-1)|`.

It targets 100% of the highest strictly positive score at the next open. If no
score is positive at month start it targets 511880.SH, then recomputes after
each subsequent close until a positive score first appears; that winner is
bought at the next open and held until the next month. 511880.SH also
participates in the ranking and can win normally. Execution uses 100-share
lots and a 0.05% one-way cost, with residuals held as cash. The comparison baseline is a fixed
35% dividend/low-vol, 40% sovereign bond, 15% short credit and 10% money-market
allocation; it does not influence the rotation selection.

## Data status

Local storage contains all eight confirmed ETFs through 2026-08-05. Each ETF
enters a backtest only on its actual first local trading date. The no-positive-
score fallback targets 511880.SH; residual cash remains possible because
execution uses 100-share lots.
