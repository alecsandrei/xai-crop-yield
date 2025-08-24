from __future__ import annotations

import collections.abc as c
import pprint
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from operator import itemgetter

import captum
import ollama
import torch

from xai_crop_yield.config import DEVICE, MODELS_DIR, RAW_DATA_DIR
from xai_crop_yield.dataset import SustainBenchCropYieldTimeseries
from xai_crop_yield.modeling.train import ConvLSTMModel

FeatureAttribution = tuple[str, float]
Attributions = list[FeatureAttribution]


class ExplainerOutput(t.TypedDict):
    method: str
    attributions: Attributions


@dataclass
class Explainer(ABC):
    model: ConvLSTMModel

    @abstractmethod
    def get_feature_mask(self, input: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def attributions_to_dict(
        self, attributions: torch.Tensor
    ) -> list[t.Any]: ...

    @abstractmethod
    def index_attributions(
        self, attributions: torch.Tensor
    ) -> torch.Tensor: ...

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


@dataclass
class MultivariateTimeseriesExplainer(Explainer):
    timestamps: c.Sequence[str]
    channel_names: c.Sequence[str]

    def get_feature_mask(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5, (
            f'Expected B, T, C, H, W shape, found {input.ndim}'
        )
        B, T, C, H, W = input.shape
        return (
            torch.arange(C * W)
            .view(1, C, 1, W)
            .expand(1, T, C, H, W)
            .to(DEVICE)
        )

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
        return attributions[:, 0, :, 0, :]


@dataclass
class TimeseriesExplainer(Explainer):
    timestamps: c.Sequence[str]

    def get_feature_mask(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5, (
            f'Expected B, T, C, H, W shape, found {input.ndim}'
        )
        B, T, C, H, W = input.shape
        return (
            torch.arange(W).view(1, 1, 1, 1, W).expand(1, T, C, H, W).to(DEVICE)
        )

    def attributions_to_dict(
        self,
        attributions: torch.Tensor,
    ) -> list[Attributions]:
        attribution_maps = []
        for batch in attributions:
            attribution_map = []
            for name, attribution in zip(self.timestamps, batch):
                attribution_map.append(
                    (name, round(float(attribution.cpu()), 4))
                )
            attribution_maps.append(attribution_map)
            attribution_map.sort(key=itemgetter(1), reverse=True)
        return attribution_maps

    def index_attributions(self, attributions: torch.Tensor) -> torch.Tensor:
        return attributions[:, 0, 0, 0]


@dataclass
class ChannelExplainer(Explainer):
    channel_names: c.Sequence[str]

    def attributions_to_dict(
        self,
        attributions: torch.Tensor,
    ) -> list[Attributions]:
        attribution_maps = []
        for batch in attributions:
            attribution_map = []
            for name, attribution in zip(self.channel_names, batch):
                attribution_map.append(
                    (name, round(float(attribution.cpu()), 4))
                )
            attribution_maps.append(attribution_map)
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
        return attributions[:, 0, :, 0, 0]


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

{pprint.pformat(self.multivariate_timeseries_explainers, indent=1)}
"""
            prompt_string = '\n'.join(
                [multivariate_timeseries_explainers_prompt, prompt_string]
            )
        return prompt_string


if __name__ == '__main__':
    model = ConvLSTMModel.load_from_checkpoint(
        MODELS_DIR / 'checkpoint.ckpt'
    ).to(DEVICE)

    index = 50
    dataset = SustainBenchCropYieldTimeseries(
        RAW_DATA_DIR, country='usa', years=list(range(2005, 2016))
    )
    data, target = dataset[index]
    data = data.to(DEVICE).unsqueeze(0)
    prediction = model(data)
    channel_explainer = ChannelExplainer(model, dataset._feature_names)
    timeseries_explainer = TimeseriesExplainer(model, dataset._timestamps)
    multivariate_timeseries_explainer = MultivariateTimeseriesExplainer(
        model, dataset._timestamps, dataset._feature_names
    )
    timeseries_explainer.feature_ablation(data)
    channel_feature_ablation = channel_explainer.feature_ablation(data)
    channel_kernel_shap = channel_explainer.kernel_shap(data)
    timeseries_feature_ablation = timeseries_explainer.feature_ablation(data)
    timeseries_kernel_shap = timeseries_explainer.kernel_shap(data)
    multivariate_feature_ablation = (
        multivariate_timeseries_explainer.feature_ablation(data)
    )
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
            channel_feature_ablation,
            channel_kernel_shap,
        ],
        timeseries_explainers=[
            timeseries_feature_ablation,
            timeseries_kernel_shap,
        ],
        multivariate_timeseries_explainers=[
            multivariate_feature_ablation,
            multivariate_kernel_shap,
        ],
    )
    prompt = story.build()
    stream = story.get_stream(prompt)
    for chunk in stream:
        print(chunk['message']['content'], flush=True, end='')
