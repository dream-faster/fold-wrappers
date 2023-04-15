from importlib.util import find_spec

from fold.models.base import Model


def wrap_transformation_if_possible(model: Model) -> Model:
    if find_spec("xgboost") is not None and __wrap_xgboost(model) is not None:
        return __wrap_xgboost(model)  # type: ignore (we already check if it's not None)
    elif find_spec("lightgbm") is not None and __wrap_lightgbm(model) is not None:
        return __wrap_xgboost(model)  # type: ignore
    elif find_spec("prophet") is not None and __wrap_prophet(model) is not None:
        return __wrap_prophet(model)  # type: ignore
    elif find_spec("sktime") is not None and __wrap_sktime(model) is not None:
        return __wrap_sktime(model)  # type: ignore
    elif (
        find_spec("statsforecast") is not None
        and __wrap_statsforecast(model) is not None
    ):
        return __wrap_statsforecast(model)  # type: ignore
    else:
        return model


def __wrap_xgboost(model):
    from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor

    from .xgboost import WrapXGB

    if (
        isinstance(model, XGBRegressor)
        or isinstance(model, XGBRFRegressor)
        or isinstance(model, XGBClassifier)
        or isinstance(model, XGBRFClassifier)
    ):
        return WrapXGB.from_model(model)
    else:
        return None


def __wrap_lightgbm(model):
    from lightgbm import LGBMClassifier, LGBMRegressor

    from .lightgbm import WrapLGBM

    if isinstance(model, LGBMClassifier) or isinstance(model, LGBMRegressor):
        return WrapLGBM.from_model(model)
    else:
        return None


def __wrap_prophet(model):
    from prophet import Prophet

    from .prophet import WrapProphet

    if isinstance(model, Prophet):
        return WrapProphet.from_model(model)
    else:
        return None


def __wrap_sktime(model):
    from sktime.forecasting.base import BaseForecaster

    from .sktime import WrapSktime

    if isinstance(model, BaseForecaster):
        return WrapSktime.from_model(model)
    else:
        return None


def __wrap_statsforecast(model):
    from statsforecast.models import _TS

    from .statsforecast import WrapStatsForecast

    if isinstance(model, _TS):
        return WrapStatsForecast.from_model(model)
    else:
        return None
