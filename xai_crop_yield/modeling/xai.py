from __future__ import annotations

import collections.abc as c
import datetime
import pprint
import typing as t
from dataclasses import dataclass
from operator import itemgetter

import captum
import matplotlib.pyplot as plt
import numpy as np
import ollama
import torch

from xai_crop_yield.config import DEVICE, MODELS_DIR, RAW_DATA_DIR
from xai_crop_yield.dataset import SustainBenchCropYieldTimeseries
from xai_crop_yield.modeling.train import ConvLSTMModel

FeatureAttribution = dict | tuple[str, float]
Attributions = list[FeatureAttribution] | torch.Tensor


class ExplainerOutput(t.TypedDict):
    method: str
    attributions: Attributions


@dataclass
class Explainer:
    model: ConvLSTMModel

    def get_feature_mask(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            'The attribution method requires an implementation of get_feature_mask'
        )

    def attributions_to_dict(self, attributions: torch.Tensor) -> list[t.Any]:
        raise NotImplementedError(
            'Tee attribution method requires an implementation of attributions_to_dict'
        )

    def index_attributions(self, attributions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            'The attribution method requires an implementation of index_attributions'
        )

    def feature_ablation(self, input: torch.Tensor) -> ExplainerOutput:
        ablation = captum.attr.FeatureAblation(self.model)
        feature_mask = self.get_feature_mask(input)
        attr = self.index_attributions(
            ablation.attribute(input, feature_mask=feature_mask)
        )
        return {
            'attributions': self.attributions_to_dict(attr),
            'method': 'Feature Ablation',
        }

    def kernel_shap(self, input: torch.Tensor) -> ExplainerOutput:
        kshap = captum.attr.KernelShap(self.model)
        feature_mask = self.get_feature_mask(input)

        attr = self.index_attributions(
            kshap.attribute(input, feature_mask=feature_mask)
        )
        return {
            'attributions': self.attributions_to_dict(attr),
            'method': 'Kernel SHAP',
        }

    def occlusion(
        self,
        input: torch.Tensor,
        sliding_window_shape: tuple[int, int] = (3, 3),
    ) -> ExplainerOutput:
        occlusion = captum.attr.Occlusion(self.model)
        attr = self.index_attributions(
            occlusion.attribute(
                input,
                sliding_window_shapes=(
                    input.shape[1],
                    input.shape[2],
                    sliding_window_shape[0],
                    sliding_window_shape[1],
                ),
            )
        )
        return {'attributions': attr, 'method': 'Occlusion'}

    def deeplift(self, input: torch.Tensor) -> ExplainerOutput:
        deeplift = captum.attr.DeepLift(self.model)
        attr = deeplift.attribute(input)
        return {'attributions': attr, 'method': 'DeepLift'}


@dataclass
class HeatmapExplainer(Explainer):
    def index_attributions(self, input: torch.Tensor) -> torch.Tensor:
        return input[:, 0, 0]

    def plot(
        self,
        attribution: torch.Tensor,
        show: bool = False,
        ax: plt.Axes | None = None,
        **kwargs,
    ) -> plt.Axes:
        assert attribution.ndim == 2
        arr = attribution.detach().cpu().numpy()
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(arr, **kwargs)
        if show:
            plt.show()
        return ax


@dataclass
class MultivariateTimeseriesExplainer(Explainer):
    timestamps: c.Sequence[str]
    channel_names: c.Sequence[str]
    timestamp_groups: c.Sequence[int] | None = None
    channel_groups: c.Sequence[int] | None = None

    def get_feature_mask(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5, (
            f'Expected B, T, C, H, W shape, found {input.ndim}'
        )
        B, T, C, H, W = input.shape
        if (
            self.timestamp_groups is not None
            and self.channel_groups is not None
        ):
            assert len(self.timestamp_groups) == W
            assert len(self.channel_groups) == C
            expandable_range = (
                (torch.Tensor(self.timestamp_groups).expand(C, W) + 1)
                * (torch.Tensor(self.channel_groups).reshape(-1, 1) + 1)
            ).view(1, C, 1, W)
        else:
            expandable_range = torch.arange(C * W).view(1, C, 1, W)
        return expandable_range.expand(1, T, C, H, W).to(torch.int).to(DEVICE)

    def attributions_to_dict(
        self,
        attributions: torch.Tensor,
    ) -> list[dict[str, Attributions]]:
        attribution_maps: list[dict[str, Attributions]] = []
        for batch in attributions:
            channel_attribution_map: dict[str, list[tuple[str, float]]] = {}
            for channel_name, channel_attributions in zip(
                self.channel_names, batch
            ):
                for timestamp, timestamp_attribution in zip(
                    self.timestamps, channel_attributions
                ):
                    prediction = round(float(timestamp_attribution.cpu()), 4)
                    channel_attribution_map.setdefault(channel_name, []).append(
                        (timestamp, prediction)
                    )
            attribution_maps.append(channel_attribution_map)
        return attribution_maps

    def index_attributions(self, attributions: torch.Tensor) -> torch.Tensor:
        timestamp_indices = list(range((attributions.shape[4])))
        channel_indices = list(range((attributions.shape[2])))
        if self.timestamp_groups is not None:
            unique = list(dict.fromkeys(self.timestamp_groups))
            timestamp_indices = [
                self.timestamp_groups.index(val) for val in unique
            ]

        if self.channel_groups is not None:
            unique = list(dict.fromkeys(self.channel_groups))
            channel_indices = [self.channel_groups.index(val) for val in unique]
        return attributions[:, 0, channel_indices, 0][:, :, timestamp_indices]


@dataclass
class TimeseriesExplainer(Explainer):
    timestamps: c.Sequence[str]
    groups: c.Sequence[int] | None = None
    sort: bool = True

    def get_feature_mask(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5, (
            f'Expected B, T, C, H, W shape, found {input.ndim}'
        )
        B, T, C, H, W = input.shape
        if self.groups is not None:
            groups = torch.Tensor(self.groups).to(torch.int)
        else:
            groups = torch.arange(W)
        return groups.view(1, 1, 1, 1, W).expand(1, T, C, H, W).to(DEVICE)

    def attributions_to_dict(
        self,
        attributions: torch.Tensor,
    ) -> list[Attributions]:
        attribution_maps: list[Attributions] = []
        for batch in attributions:
            attribution_map = []
            for name, attribution in zip(self.timestamps, batch):
                attribution_map.append(
                    (name, round(float(attribution.cpu()), 4))
                )
            attribution_maps.append(attribution_map)
            if self.sort:
                attribution_map.sort(key=itemgetter(1), reverse=True)
        return attribution_maps

    def index_attributions(self, attributions: torch.Tensor) -> torch.Tensor:
        indices = list(range((attributions.shape[4])))
        if self.groups is not None:
            unique = list(dict.fromkeys(self.groups))
            indices = [self.groups.index(val) for val in unique]
        return attributions[:, 0, 0, 0, indices]


@dataclass
class ChannelExplainer(Explainer):
    channel_names: c.Sequence[str]
    groups: c.Sequence[int] | None = None
    sort: bool = True

    def attributions_to_dict(
        self,
        attributions: torch.Tensor,
    ) -> list[Attributions]:
        attribution_maps: list[Attributions] = []
        for batch in attributions:
            attribution_map = []
            for name, attribution in zip(self.channel_names, batch):
                attribution_map.append(
                    (name, round(float(attribution.cpu()), 4))
                )
            attribution_maps.append(attribution_map)
            if self.sort:
                attribution_map.sort(key=itemgetter(1), reverse=True)
        return attribution_maps

    def get_feature_mask(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5, (
            f'Expected B, T, C, H, W shape, found {input.ndim}'
        )
        B, T, C, H, W = input.shape
        return (
            torch.arange(C).view(1, 1, C, 1, 1).expand(1, T, C, H, W).to(DEVICE)
        )

    def index_attributions(self, attributions: torch.Tensor) -> torch.Tensor:
        indices = list(range((attributions.shape[2])))
        if self.groups is not None:
            unique = list(dict.fromkeys(self.groups))
            indices = [self.groups.index(val) for val in unique]
        return attributions[:, 0, indices, 0, 0]


@dataclass
class AttributionStory:
    dataset_description: str
    input_description: str
    output_description: str
    county: str
    prediction: float
    target: float
    channel_explainers: c.Sequence[ExplainerOutput] | None = None
    timeseries_explainers: c.Sequence[ExplainerOutput] | None = None
    multivariate_timeseries_explainers: c.Sequence[ExplainerOutput] | None = (
        None
    )

    def get_stream(self, prompt: str):
        stream = ollama.chat(
            model='gemma3',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            stream=True,
        )
        return stream

    def build(self) -> str:
        prompt_string = f"""
{self.dataset_description}
{self.input_description}
{self.output_description}

The chosen model was a ConvLSTM. The input dimension shape is represented by B, T, C, H, W.

For the {self.county} county the prediction was {self.prediction:.3f} with a ground
truth value of {self.target:.3f}. XAI methods were employed to explain the prediction and 
a summary of the results will be provided in JSON. Important to note that the channel attributions and the
timeseries attributions are not related! ALl of the results are already sorted in descending order.

Can you come up with a plausible, fluent story as to why the model could have
predicted this outcome, based on the most influential positive and most influential
negative attribution values? Focus on the features with the highest absolute
attribution values. Try to explain the most important feature values in this story, as
well as potential interactions that fit the story. No need to enumerate individual
features outside of the story. Conclude with a short summary of why this
classification may have occurred. Limit your answer to 8 sentences. Do not mention model details
or technical details, make it user friendly.
"""
        if self.channel_explainers:
            channel_explainers_prompt = f"""
Channel explainer results:

{pprint.pformat(self.channel_explainers, indent=1)}
"""
            prompt_string = '\n'.join(
                [prompt_string, channel_explainers_prompt]
            )
        if self.timeseries_explainers:
            timeseries_explainers_prompt = f"""
Timeseries explainer results:

{pprint.pformat(self.timeseries_explainers, indent=1)}
"""
            prompt_string = '\n'.join(
                [prompt_string, timeseries_explainers_prompt]
            )
        if self.multivariate_timeseries_explainers:
            multivariate_timeseries_explainers_prompt = f"""
Multivariate timeseries explainer results:
According to USDA, for soybean, Planting season ends in July,
Mid-Season ends in September and the Harvest season ends in November.

{pprint.pformat(self.multivariate_timeseries_explainers, indent=1)}
"""
            prompt_string = '\n'.join(
                [prompt_string, multivariate_timeseries_explainers_prompt]
            )
        return prompt_string


def get_modis_bands_groups(
    dataset: SustainBenchCropYieldTimeseries,
) -> tuple[list[int], dict[int, str]]:
    group_labels = {0: 'Visible', 1: 'NIR', 2: 'SWIR', 3: 'LST'}
    band_label_mapper = [0, 1, 0, 0, 1, 2, 2, 3, 3]
    return (band_label_mapper, group_labels)


def get_crop_calendar_groups(
    timestamps: list[str],
) -> tuple[list[int], dict[int, str]]:
    dummy_year = 1970
    soybean_calendar_labels = {0: 'Plant', 1: 'Mid-Season', 2: 'Harvest'}
    soybean_calendar = [
        datetime.datetime(dummy_year, 7, 1).timestamp(),  # start of mid-season
        datetime.datetime(
            dummy_year, 9, 1
        ).timestamp(),  # start of harvest season
    ]

    datetimes = []
    for timestamp in timestamps:
        as_datetime = datetime.datetime.strptime(timestamp, '%B-%d')
        as_datetime = datetime.datetime(
            dummy_year, as_datetime.month, as_datetime.day
        )
        datetimes.append(as_datetime.timestamp())

    groups = np.digitize(datetimes, soybean_calendar).tolist()

    return (groups, soybean_calendar_labels)


if __name__ == '__main__':
    model = ConvLSTMModel.load_from_checkpoint(
        MODELS_DIR / 'checkpoint.ckpt'
    ).to(DEVICE)

    index = 52
    dataset = SustainBenchCropYieldTimeseries(
        RAW_DATA_DIR, country='usa', years=list(range(2005, 2016))
    )
    data, target = dataset[index]
    data = data.to(DEVICE).unsqueeze(0)
    target = target.to(DEVICE).view(-1, 1)
    prediction = model(data)
    crop_calendar_groups, crop_calendar_labels = get_crop_calendar_groups(
        dataset._timestamps
    )
    # explainer = HeatmapExplainer(model)
    # attributions = explainer.deeplift(data)
    # explainer.plot(attributions['attributions'][0])
    channel_explainer = ChannelExplainer(model, dataset._feature_names)
    timeseries_explainer = TimeseriesExplainer(model, dataset._timestamps)
    timeseries_cropcalendar_explainer = TimeseriesExplainer(
        model, list(crop_calendar_labels.values()), groups=crop_calendar_groups
    )
    cropcalendar_attributions = (
        timeseries_cropcalendar_explainer.feature_ablation(data)
    )
    modis_band_groups, modis_group_labels = get_modis_bands_groups(dataset)
    multivariate_timeseries_explainer = MultivariateTimeseriesExplainer(
        model,
        list(crop_calendar_labels.values()),
        list(modis_group_labels.values()),
        crop_calendar_groups,
        modis_band_groups,
    )
    channel_feature_ablation = channel_explainer.feature_ablation(data)
    channel_kernel_shap = channel_explainer.kernel_shap(data)
    timeseries_feature_ablation = timeseries_explainer.feature_ablation(data)
    timeseries_kernel_shap = timeseries_explainer.kernel_shap(data)
    multivariate_feature_ablation = (
        multivariate_timeseries_explainer.feature_ablation(data)
    )
    breakpoint()
    multivariate_kernel_shap = multivariate_timeseries_explainer.kernel_shap(
        data
    )
    story = AttributionStory(
        dataset._dataset_description,
        dataset._input_description,
        dataset._output_description,
        county=str(dataset.locations[index]),
        target=round(float(target.cpu()), 4),
        prediction=round(float(prediction.detach().cpu()), 4),
        channel_explainers=[
            # channel_feature_ablation,
            # channel_kernel_shap,
        ],
        timeseries_explainers=[
            # timeseries_feature_ablation,
            # timeseries_kernel_shap,
        ],
        multivariate_timeseries_explainers=[
            multivariate_feature_ablation,
            # multivariate_kernel_shap,
        ],
    )
    prompt = story.build()
    stream = story.get_stream(prompt)
    print(prompt)
    for chunk in stream:
        print(chunk['message']['content'], flush=True, end='')
