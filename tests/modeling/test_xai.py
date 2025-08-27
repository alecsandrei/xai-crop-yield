from __future__ import annotations

from functools import cache

from xai_crop_yield.config import DEVICE
from xai_crop_yield.dataset import get_dataset
from xai_crop_yield.modeling.train import get_model
from xai_crop_yield.modeling.xai import (
    ChannelExplainer,
    HeatmapExplainer,
    MultivariateTimeseriesExplainer,
    TimeseriesExplainer,
    get_crop_calendar_groups,
)

test_index = 50
model = get_model()
dataset = get_dataset()


@cache
def get_data():
    data, target = dataset[test_index]
    data = data.to(DEVICE).unsqueeze(0)
    return data


def test_channel_explainer():
    channel_explainer = ChannelExplainer(model, dataset._feature_names)
    feature_ablation = channel_explainer.feature_ablation(get_data())
    kernel_shap = channel_explainer.kernel_shap(get_data())


def test_timeseries_explainer():
    timeseries_explainer = TimeseriesExplainer(model, dataset._feature_names)
    feature_ablation = timeseries_explainer.feature_ablation(get_data())
    kernel_shap = timeseries_explainer.kernel_shap(get_data())


def test_timeseries_cropcalendar_explainer():
    groups, labels = get_crop_calendar_groups(dataset._timestamps)
    timeseries_explainer = TimeseriesExplainer(
        model, list(labels.values()), groups
    )
    feature_ablation = timeseries_explainer.feature_ablation(get_data())
    kernel_shap = timeseries_explainer.kernel_shap(get_data())


def test_multivariate_timeseries_explainer():
    multivariate_timeseries_explainer = MultivariateTimeseriesExplainer(
        model, dataset._timestamps, dataset._feature_names
    )
    feature_ablation = multivariate_timeseries_explainer.feature_ablation(
        get_data()
    )
    kernel_shap = multivariate_timeseries_explainer.kernel_shap(get_data())


def test_heatmap_explainer():
    explainer = HeatmapExplainer(model)
    deeplift = explainer.deeplift(get_data())
    occlusion = explainer.occlusion(get_data())
