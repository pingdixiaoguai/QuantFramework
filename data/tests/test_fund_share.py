from __future__ import annotations

from datetime import date

import pandas as pd

from data.fund_share import fetch_fund_share


def test_fetch_fund_share_returns_sorted_numeric_point_in_time_series() -> None:
    class FakePro:
        def fund_share(self, **kwargs):
            assert kwargs == {
                "ts_code": "159915.SZ",
                "start_date": "20260801",
                "end_date": "20260803",
            }
            return pd.DataFrame(
                {
                    "trade_date": ["20260803", "20260801", "20260802"],
                    "fd_share": ["103", "100", "101"],
                }
            )

    result = fetch_fund_share(
        "159915.SZ",
        date(2026, 8, 1),
        date(2026, 8, 3),
        pro=FakePro(),
    )

    assert list(result.index) == list(pd.to_datetime(["20260801", "20260802", "20260803"]))
    assert result.tolist() == [100.0, 101.0, 103.0]
