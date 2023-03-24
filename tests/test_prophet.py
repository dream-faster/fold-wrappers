from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data
from prophet import Prophet

from fold_models.prophet import WrapProphet


def test_prophet() -> None:
    X, y = generate_sine_wave_data(cycles=100, length=1200, freq="D")

    splitter = ExpandingWindowSplitter(initial_train_window=0.5, step=0.05)
    transformations = WrapProphet.from_model(Prophet())

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (y.squeeze()[pred.index][:-1] - pred.squeeze()[:-1]).abs().sum() < 20


test_prophet()
