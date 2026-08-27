import numpy as np
import pandas as pd

from factors.quality_momentum import compute as quality_momentum


def test_qm_zero_threshold_has_same_sign_as_log_return() -> None:
    dates = pd.bdate_range("2024-01-02", periods=100)
    prices = 100.0 * np.exp(
        np.cumsum(np.sin(np.arange(100) / 7.0) * 0.01 + 0.001)
    )
    frame = pd.DataFrame({"date": dates, "close": prices})
    qm = quality_momentum(frame, {"window": 40})
    signed_return = np.log(frame["close"]).diff(40)
    signed_return.index = dates
    finite = qm.notna() & signed_return.notna()

    assert finite.any()
    assert qm.loc[finite].gt(0.0).equals(signed_return.loc[finite].gt(0.0))
