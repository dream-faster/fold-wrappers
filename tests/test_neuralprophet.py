from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data
from neuralprophet import NeuralProphet

from fold_models.neuralprophet import NeuralProphetWrapper


def test_neuralprophet() -> None:
    X = generate_sine_wave_data(resolution=400)
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=300, step=50)
    transformations = NeuralProphetWrapper(NeuralProphet(yearly_seasonality=False))

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (y.squeeze()[pred.index][:-1] - pred.squeeze()[:-1]).abs().sum() < 20


test_neuralprophet()
