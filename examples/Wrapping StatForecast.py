from fold import ExpandingWindowSplitter, train_evaluate
from fold.utils.dataset import get_preprocessed_dataset
from statsforecast.models import ARIMA

from fold_models import WrapStatsForecast

X, y = get_preprocessed_dataset(
    "weather/historical_hourly_la", target_col="temperature", shorten=1000
)
model = WrapStatsForecast(
    model_class=ARIMA,
    init_args={"order": (1, 0, 0)},
    use_exogenous=False,
    online_mode=False,
)
splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=50)

scorecard, predictions, trained_pipeline = train_evaluate(model, X, y, splitter)

model_init = WrapStatsForecast.from_model(
    ARIMA(order=(1, 0, 0)), use_exogenous=False, online_mode=False
)
scorecard, predictions, trained_pipeline = train_evaluate(model_init, X, y, splitter)
