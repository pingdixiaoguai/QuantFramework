# 20-day shape signal diagnostic

- Run date: 2026-05-20
- Assets: 510300.SH, 159915.SZ, 513100.SH, 518880.SH
- Signal dates: 2014-01-01 to 2026-05-19
- Prices: local qfq-adjusted open/close from `data.store.query()`.
- ER: recovered from `factors.quality_momentum.compute(window=20)` divided by the same 20-day close momentum.
- Forward return: `open[t+6] / open[t+1] - 1`.
- Standardization: percentile rank within each asset for ER, momentum, conv, and com before tercile grouping.
- Runtime: 2.17s

## Alignment samples

| asset     | window_start | window_end | t_plus_1   | t_plus_6   | fwd      |
| --------- | ------------ | ---------- | ---------- | ---------- | -------- |
| 510300.SH | 2013-12-04   | 2014-01-02 | 2014-01-03 | 2014-01-10 | -3.7744% |
| 159915.SZ | 2014-02-20   | 2014-03-20 | 2014-03-21 | 2014-03-28 | -3.4232% |
| 513100.SH | 2026-04-08   | 2026-05-11 | 2026-05-12 | 2026-05-19 | -1.9066% |

## Sample counts

- Full overlapping sample rows: 11998
- Every-5-trading-day sample rows: 2400
- Full ER x conv cell count range: 1265-1404
- Full ER x com cell count range: 1046-1703
- Full ER x mom x conv cell count range: 21-1016
- Full ER x mom x com cell count range: 25-1150
- Every-5 ER x conv cell count range: 258-280
- Every-5 ER x com cell count range: 194-354
- Every-5 ER x mom x conv cell count range: 3-213
- Every-5 ER x mom x com cell count range: 5-238

## Objective observations

- Full ER x conv high-minus-low spreads: ER=low: high-low mean -0.6120%, median -0.0451%, min count 1292; ER=mid: high-low mean -0.3485%, median 0.1477%, min count 1265; ER=high: high-low mean -0.0726%, median -0.0099%, min count 1331.
- Full ER x com high-minus-low spreads: ER=low: high-low mean 0.2995%, median -0.0021%, min count 1067; ER=mid: high-low mean 0.0788%, median -0.0590%, min count 1230; ER=high: high-low mean 0.4153%, median 0.4035%, min count 1046.
- Every-5 ER x conv high-minus-low spreads: ER=low: high-low mean -0.2242%, median 0.0585%, min count 266; ER=mid: high-low mean -0.6008%, median 0.0894%, min count 260; ER=high: high-low mean -0.1663%, median -0.0982%, min count 258.
- Every-5 ER x com high-minus-low spreads: ER=low: high-low mean 0.5961%, median 0.1787%, min count 222; ER=mid: high-low mean 0.2434%, median 0.0855%, min count 221; ER=high: high-low mean 0.2925%, median 0.2496%, min count 194.
- Full ER x mom x conv largest absolute high-minus-low mean spreads: ER=low, mom=high: high-low mean -1.3218%, median -1.3654%, min count 32; ER=high, mom=mid: high-low mean -0.9813%, median -0.7516%, min count 21; ER=mid, mom=low: high-low mean -0.8878%, median 0.0010%, min count 515.
- Full ER x mom x com largest absolute high-minus-low mean spreads: ER=low, mom=high: high-low mean -1.5130%, median -1.9256%, min count 25; ER=high, mom=low: high-low mean 0.4566%, median 0.7226%, min count 287; ER=low, mom=mid: high-low mean 0.4322%, median 0.1198%, min count 801.
- Every-5 ER x mom x conv largest absolute high-minus-low mean spreads: ER=high, mom=mid: high-low mean -1.1848%, median -1.1005%, min count 4; ER=mid, mom=low: high-low mean -1.1795%, median 0.0695%, min count 104; ER=mid, mom=high: high-low mean -0.3826%, median 0.5354%, min count 82.
- Every-5 ER x mom x com largest absolute high-minus-low mean spreads: ER=low, mom=mid: high-low mean 0.8431%, median 0.2939%, min count 166; ER=mid, mom=low: high-low mean 0.7827%, median 0.0749%, min count 111; ER=high, mom=low: high-low mean 0.6469%, median 0.6337%, min count 51.

## Full sample: ER then conv

| er_group | conv_group | mean     | median  | count |
| -------- | ---------- | -------- | ------- | ----- |
| low      | low        | 0.3495%  | 0.1875% | 1292  |
| low      | mid        | 0.0916%  | 0.2147% | 1302  |
| low      | high       | -0.2625% | 0.1424% | 1404  |
| mid      | low        | 0.3233%  | 0.1508% | 1346  |
| mid      | mid        | 0.0310%  | 0.1170% | 1388  |
| mid      | high       | -0.0253% | 0.2985% | 1265  |
| high     | low        | 0.6721%  | 0.4954% | 1360  |
| high     | mid        | 0.4970%  | 0.4170% | 1310  |
| high     | high       | 0.5995%  | 0.4855% | 1331  |

## Full sample: ER then com

| er_group | com_group | mean     | median  | count |
| -------- | --------- | -------- | ------- | ----- |
| low      | low       | -0.1402% | 0.1729% | 1557  |
| low      | mid       | 0.1824%  | 0.1860% | 1374  |
| low      | high      | 0.1593%  | 0.1708% | 1067  |
| mid      | low       | 0.0636%  | 0.1858% | 1395  |
| mid      | mid       | 0.1328%  | 0.2225% | 1374  |
| mid      | high      | 0.1424%  | 0.1268% | 1230  |
| high     | low       | 0.3375%  | 0.2594% | 1046  |
| high     | mid       | 0.5815%  | 0.4190% | 1252  |
| high     | high      | 0.7527%  | 0.6629% | 1703  |

## Every-5 sample: ER then conv

| er_group | conv_group | mean     | median  | count |
| -------- | ---------- | -------- | ------- | ----- |
| low      | low        | 0.2532%  | 0.1694% | 266   |
| low      | mid        | 0.0381%  | 0.1847% | 277   |
| low      | high       | 0.0290%  | 0.2279% | 280   |
| mid      | low        | 0.2361%  | 0.0594% | 260   |
| mid      | mid        | 0.0557%  | 0.0561% | 259   |
| mid      | high       | -0.3647% | 0.1487% | 263   |
| high     | low        | 0.8014%  | 0.5757% | 277   |
| high     | mid        | 0.6028%  | 0.4675% | 260   |
| high     | high       | 0.6351%  | 0.4775% | 258   |

## Every-5 sample: ER then com

| er_group | com_group | mean     | median  | count |
| -------- | --------- | -------- | ------- | ----- |
| low      | low       | -0.1140% | 0.1679% | 322   |
| low      | mid       | 0.0563%  | 0.1754% | 279   |
| low      | high      | 0.4821%  | 0.3466% | 222   |
| mid      | low       | -0.1306% | 0.0367% | 281   |
| mid      | mid       | -0.0299% | 0.1401% | 280   |
| mid      | high      | 0.1129%  | 0.1222% | 221   |
| high     | low       | 0.4682%  | 0.4306% | 194   |
| high     | mid       | 0.7387%  | 0.4452% | 247   |
| high     | high      | 0.7607%  | 0.6803% | 354   |

## ER + momentum + shape tables

CSV outputs:
- `2026-05-20_shape_signal_diagnostic_every_5_er_com.csv`
- `2026-05-20_shape_signal_diagnostic_every_5_er_conv.csv`
- `2026-05-20_shape_signal_diagnostic_every_5_er_mom_com.csv`
- `2026-05-20_shape_signal_diagnostic_every_5_er_mom_conv.csv`
- `2026-05-20_shape_signal_diagnostic_full_er_com.csv`
- `2026-05-20_shape_signal_diagnostic_full_er_conv.csv`
- `2026-05-20_shape_signal_diagnostic_full_er_mom_com.csv`
- `2026-05-20_shape_signal_diagnostic_full_er_mom_conv.csv`
