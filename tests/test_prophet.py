from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data
from prophet import Prophet

from fold_models.prophet import WrapProphet


def test_neuralprophet() -> None:
    X, y = generate_sine_wave_data(length=400)

    splitter = ExpandingWindowSplitter(initial_train_window=300, step=50)
    transformations = WrapProphet.from_model(Prophet())

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (y.squeeze()[pred.index][:-1] - pred.squeeze()[:-1]).abs().sum() < 20


test_neuralprophet()
