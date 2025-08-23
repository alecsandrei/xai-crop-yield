from __future__ import annotations

import collections.abc as c
import pprint
import typing as t
from dataclasses import dataclass

import captum
import ollama
import shap
import torch
from torch.utils.data import DataLoader

from xai_crop_yield.config import DEVICE, MODELS_DIR, RAW_DATA_DIR
from xai_crop_yield.dataset import SustainBenchCropYieldTimeseries
from xai_crop_yield.modeling.train import ConvLSTMModel


def gradient_explainer(model: ConvLSTMModel, loader: DataLoader):
    batch = next(iter(loader))
    data, _ = batch
    data = data.to(DEVICE)
    background = data[:4]
    test = data[4:]
    explainer = shap.GradientExplainer(model, data)
    shap_values = explainer.shap_values(data[:1])
    explainer.explain_row
    breakpoint()


def captum_explainer(model: ConvLSTMModel, loader: DataLoader):
    # kernel_shap = captum.attr.KernelShap(model)
    permutation = captum.attr.FeatureAblation(model)
    batch = next(iter(loader))
    data, _ = batch
    data = data.to(DEVICE)
    B, T, C, H, W = data.shape
    feature_mask = (
        torch.arange(C).view(1, 1, C, 1, 1).expand(1, T, C, H, W).to(DEVICE)
    )
    attr = permutation.attribute(data, feature_mask=feature_mask)
    breakpoint()


class FeatureAttribution(t.TypedDict):
    name: str
    attribution: float


class ChannelExplainerOutput(t.TypedDict):
    method: str
    channel_attribution: c.Sequence[FeatureAttribution]


@dataclass
class ChannelExplainer:
    model: ConvLSTMModel
    channel_names: c.Sequence[str]

    def attributions_to_dict(
        self,
        attributions: torch.Tensor,
    ) -> list[FeatureAttribution]:
        attribution_maps = []
        for batch in attributions:
            attribution_map = {}
            for name, attribution in zip(self.channel_names, batch):
                attribution_map[name] = round(float(attribution.cpu()), 4)
            attribution_maps.append(attribution_map)
        return attribution_maps

    def get_feature_mask(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5, (
            f'Expected B, T, C, H, W shape, found {input.ndim}'
        )
        B, T, C, H, W = input.shape
        return (
            torch.arange(C).view(1, 1, C, 1, 1).expand(1, T, C, H, W).to(DEVICE)
        )

    def feature_ablation(self, input: torch.Tensor) -> ChannelExplainerOutput:
        ablation = captum.attr.FeatureAblation(self.model)
        feature_mask = self.get_feature_mask(input)
        attr = ablation.attribute(data, feature_mask=feature_mask)
        return {
            'channel_attribution': self.attributions_to_dict(
                attr.view(attr.shape[0], -1).unique(dim=1)
            ),
            'method': 'Feature Ablation',
        }

    def kernel_shap(self, input: torch.Tensor) -> ChannelExplainerOutput:
        kshap = captum.attr.KernelShap(self.model)
        feature_mask = self.get_feature_mask(input)

        attr = kshap.attribute(data, feature_mask=feature_mask)
        return {
            'channel_attribution': self.attributions_to_dict(
                attr.view(attr.shape[0], -1).unique(dim=1)
            ),
            'method': 'Kernel SHAP',
        }


@dataclass
class AttributionStory:
    dataset_description: str
    input_description: str
    output_description: str
    county: str
    prediction: float
    target: float
    channel_explainers: c.Sequence[ChannelExplainerOutput]

    def get_response(self):
        prompt = self.build()
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
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)

    def build(self) -> str:
        prompt_string = f"""
{self.dataset_description}
{self.input_description}
{self.output_description}

The chosen model was a ConvLSTM. The input dimension shape is represented by B, T, C, H, W.

For the {self.county} county the prediction was {self.prediction} with a ground
truth value of {self.target}. XAI methods were employed to explain the prediction and below
a summary of the results will be provided in JSON.

Can you come up with a plausible, fluent story as to why the model could have
predicted this outcome, based on the most influential positive and most influential
negative attribution values? Focus on the features with the highest absolute
attribution values. Try to explain the most important feature values in this story, as
well as potential interactions that fit the story. No need to enumerate individual
features outside of the story. Conclude with a short summary of why this
classification may have occurred. Limit your answer to 8 sentences. Do not mention model details
or technical details, make it user friendly.

Channel explainer results:

{pprint.pformat(self.channel_explainers, indent=1)}"""

        return prompt_string


if __name__ == '__main__':
    model = ConvLSTMModel.load_from_checkpoint(
        MODELS_DIR / 'checkpoint.ckpt'
    ).to(DEVICE)

    index = 50
    dataset = SustainBenchCropYieldTimeseries(
        RAW_DATA_DIR, country='usa', years=list(range(2005, 2016))
    )
    dataset._load()
    data, target = dataset[index]
    data = data.to(DEVICE).unsqueeze(0)
    prediction = model(data)
    channel_explainer = ChannelExplainer(model, dataset._feature_names)
    feature_ablation_attributions = channel_explainer.feature_ablation(data)
    kernel_shap_attributions = channel_explainer.kernel_shap(data)
    AttributionStory(
        dataset._dataset_description,
        dataset._input_description,
        dataset._output_description,
        county=str(dataset.locations[index]),
        target=round(float(target.cpu()), 4),
        prediction=round(float(prediction.detach().cpu()), 4),
        channel_explainers=[
            feature_ablation_attributions,
            kernel_shap_attributions,
        ],
    ).get_response()
    breakpoint()
