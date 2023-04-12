from fold.loop import train_backtest
from fold.splitters import ExpandingWindowSplitter, SingleWindowSplitter
from fold.utils.tests import generate_monotonous_data, generate_sine_wave_data
from lightgbm import LGBMRegressor

from fold_models.lightgbm import WrapLGBM


def test_lgbm() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapLGBM.from_model(LGBMRegressor())

    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (y.squeeze()[pred.index] - pred.squeeze()).abs().sum() < 20


def test_automatic_wrapping_lgbm() -> None:
    X, y = generate_monotonous_data()
    train_backtest(
        LGBMRegressor(),
        X,
        y,
        splitter=SingleWindowSplitter(0.5),
    )


def test_lgbm_init_with_args() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapLGBM(LGBMRegressor, {"n_estimators": 100})

    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (y.squeeze()[pred.index] - pred.squeeze()).abs().sum() < 20
