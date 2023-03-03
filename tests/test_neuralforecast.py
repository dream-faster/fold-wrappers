from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import OnlyPredictions
from fold.utils.tests import generate_all_zeros
from neuralforecast.models import NBEATS

from fold_models.neuralforecast import UnivariateNeuralForecast


def test_sequential() -> None:

    X = generate_all_zeros(1000)
    y = X.squeeze()

    step = 400
    splitter = ExpandingWindowSplitter(train_window_size=400, step=step)

    transformations = [
        UnivariateNeuralForecast(NBEATS(input_size=step, h=1, max_epochs=50)),
        OnlyPredictions(),
    ]
    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred.squeeze() == y[pred.index]).all()


#%%
# from neuralforecast import NeuralForecast
# from neuralforecast.models import NBEATS, NHITS
# from neuralforecast.utils import AirPassengersDF

# # Split data and declare panel dataset
# Y_df = AirPassengersDF
# Y_train_df = Y_df[Y_df.ds <= "1959-12-31"]  # 132 train
# Y_test_df = Y_df[Y_df.ds > "1959-12-31"]  # 12 test

# # Fit and predict with N-BEATS and N-HiTS models
# horizon = len(Y_test_df)
# models = [
#     NBEATS(input_size=2 * horizon, h=horizon, max_epochs=50),
#     NHITS(input_size=2 * horizon, h=horizon, max_epochs=50),
# ]
# nforecast = NeuralForecast(models=models, freq="M")
# nforecast.fit(df=Y_train_df)
# Y_hat_df = nforecast.predict().reset_index()

# # %%
