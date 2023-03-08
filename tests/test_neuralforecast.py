from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import OnlyPredictions
from fold.utils.tests import generate_all_zeros
from neuralforecast.models import NBEATS

from fold_models.neuralforecast import UnivariateNeuralForecast


def test_neuralforecast_univariate() -> None:

    X = generate_all_zeros(1000)
    y = X.squeeze()

    step = 10
    splitter = ExpandingWindowSplitter(train_window_size=400, step=step)

    transformations = UnivariateNeuralForecast(
        NBEATS(input_size=100, h=step, max_epochs=50)
    )
    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred.squeeze() == y[pred.index]).all()
