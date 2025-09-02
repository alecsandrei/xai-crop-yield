from __future__ import annotations

import collections.abc as c
import datetime
import pprint
import typing as t
from abc import ABC, abstractmethod
from collections import UserList
from dataclasses import asdict, dataclass, fields
from enum import Enum

import captum
import matplotlib.pyplot as plt
import numpy as np
import ollama
import pandas as pd
import pydantic
import torch

from xai_crop_yield.config import DEVICE, MODELS_DIR, RAW_DATA_DIR
from xai_crop_yield.dataset import SustainBenchCropYieldTimeseries

if t.TYPE_CHECKING:
    from xai_crop_yield.modeling.train import ConvLSTMModel


class ExplainerOutput(t.TypedDict):
    explainer: str
    method: str
    attributions: list[Features]


@dataclass(frozen=True)
class Feature(ABC):
    attribution: float

    @abstractmethod
    def name(self) -> str: ...


class Features[T: Feature](UserList[T]):
    def __init__(self, initlist: None | c.Iterable[T] = None):
        super().__init__(initlist)

    def as_df(self) -> pd.DataFrame:
        cols = [f.name for f in fields(self.data[0])]
        records = [asdict(feature) for feature in self.data]
        return pd.DataFrame(records, columns=cols)

    def sort_by_attribution(self, reverse: bool = False):
        super().sort(
            key=lambda feature: abs(feature.attribution), reverse=reverse
        )


@dataclass(frozen=True)
class MultivariateTimeseriesFeature(Feature):
    timestamp: str
    channel: str

    def name(self) -> str:
        return f'{self.timestamp}, {self.channel}'


@dataclass(frozen=True)
class ChannelFeature(Feature):
    channel: str

    def name(self) -> str:
        return self.channel


@dataclass(frozen=True)
class TimeseriesFeature(Feature):
    timestamp: str

    def name(self) -> str:
        return self.timestamp


@dataclass
class Explainer[T: Feature]:
    model: ConvLSTMModel

    def get_feature_mask(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            'The attribution method requires an implementation of get_feature_mask'
        )

    def build_attribution_container(
        self, attributions: torch.Tensor
    ) -> list[Features[T]]:
        raise NotImplementedError(
            'The attribution method requires an implementation of build_attribution_container'
        )

    def index_attributions(self, attributions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            'The attribution method requires an implementation of index_attributions'
        )

    def apply_feature_mask(self, attributions: torch.Tensor) -> torch.Tensor:
        try:
            feature_mask = self.get_feature_mask(attributions)
        except NotImplementedError:
            out = attributions
        else:
            n_classes = feature_mask.max()
            out = torch.empty(attributions.shape).to(attributions.device)
            for k in range(n_classes):
                mask = (feature_mask == k).expand(attributions.shape)
                out[mask] = attributions[mask].sum()
        return out

    def feature_ablation(self, input: torch.Tensor) -> ExplainerOutput:
        ablation = captum.attr.FeatureAblation(self.model)
        feature_mask = self.get_feature_mask(input)
        attr = self.index_attributions(
            ablation.attribute(input, feature_mask=feature_mask)
        )
        return {
            'explainer': self.__class__.__name__,
            'attributions': self.build_attribution_container(attr),
            'method': 'Feature Ablation',
        }

    def kernel_shap(self, input: torch.Tensor) -> ExplainerOutput:
        kshap = captum.attr.KernelShap(self.model)
        feature_mask = self.get_feature_mask(input)

        attr = self.index_attributions(
            kshap.attribute(input, feature_mask=feature_mask)
        )
        return {
            'explainer': self.__class__.__name__,
            'attributions': self.build_attribution_container(attr),
            'method': 'Kernel SHAP',
        }

    def integrated_gradients(self, input: torch.Tensor) -> ExplainerOutput:
        ig = captum.attr.IntegratedGradients(self.model)
        attr = self.index_attributions(
            self.apply_feature_mask(ig.attribute(input))
        )

        return {
            'explainer': self.__class__.__name__,
            'attributions': self.build_attribution_container(attr),
            'method': 'Integrated Gradients',
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
        return {
            'explainer': self.__class__.__name__,
            'attributions': attr,
            'method': 'Occlusion',
        }

    def deeplift(self, input: torch.Tensor) -> ExplainerOutput:
        deeplift = captum.attr.DeepLift(self.model)
        attr = deeplift.attribute(input)
        return {
            'explainer': self.__class__.__name__,
            'attributions': attr,
            'method': 'DeepLift',
        }


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
class MultivariateTimeseriesExplainer(Explainer[MultivariateTimeseriesFeature]):
    timestamps: c.Sequence[str]
    channel_names: c.Sequence[str]
    timestamp_groups: c.Sequence[int] | None = None
    channel_groups: c.Sequence[int] | None = None
    sort: bool = True

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
                - 1
            ).view(1, C, 1, W)
        else:
            expandable_range = torch.arange(C * W).view(1, C, 1, W)
        return expandable_range.expand(1, T, C, H, W).to(torch.int).to(DEVICE)

    def build_attribution_container(
        self, attributions: torch.Tensor, sort: bool = True
    ) -> list[Features]:
        features_list = []
        for batch in attributions:
            features = Features[MultivariateTimeseriesFeature]()
            for channel_name, channel_attributions in zip(
                self.channel_names, batch
            ):
                for timestamp, timestamp_attribution in zip(
                    self.timestamps, channel_attributions
                ):
                    attribution = round(float(timestamp_attribution.cpu()), 4)
                    features.append(
                        MultivariateTimeseriesFeature(
                            attribution, timestamp, channel_name
                        )
                    )

            if self.sort:
                features.sort_by_attribution(reverse=True)
            features_list.append(features)
        return features_list

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
class TimeseriesExplainer(Explainer[TimeseriesFeature]):
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

    def build_attribution_container(
        self,
        attributions: torch.Tensor,
    ) -> list[Features[TimeseriesFeature]]:
        features_list = []
        for batch in attributions:
            features = Features[TimeseriesFeature]()
            for name, attribution in zip(self.timestamps, batch):
                features.append(
                    TimeseriesFeature(round(float(attribution.cpu()), 4), name)
                )
            features_list.append(features)
            if self.sort:
                features.sort_by_attribution(reverse=True)
        return features_list

    def index_attributions(self, attributions: torch.Tensor) -> torch.Tensor:
        indices = list(range((attributions.shape[4])))
        if self.groups is not None:
            unique = list(dict.fromkeys(self.groups))
            indices = [self.groups.index(val) for val in unique]
        return attributions[:, 0, 0, 0, indices]


@dataclass
class ChannelExplainer(Explainer[ChannelFeature]):
    channel_names: c.Sequence[str]
    groups: c.Sequence[int] | None = None
    sort: bool = True

    def build_attribution_container(
        self,
        attributions: torch.Tensor,
    ) -> list[Features[ChannelFeature]]:
        features_list = []
        for batch in attributions:
            features = Features[ChannelFeature]()
            for name, attribution in zip(self.channel_names, batch):
                features.append(
                    ChannelFeature(round(float(attribution.cpu()), 4), name)
                )
            features_list.append(features)
            if self.sort:
                features.sort_by_attribution(reverse=True)
        return features_list

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


class Models(Enum):
    LLAMA31 = 'llama3.1:8b'
    GEMMA3 = 'gemma3'


@dataclass
class BaseRequester:
    dataset_description: str
    input_description: str
    output_description: str
    county: str
    prediction: float

    def get_chat(self, prompt: str, **kwargs):
        if 'model' not in kwargs:
            if 'format' in kwargs:
                kwargs['model'] = Models.LLAMA31.value
            else:
                kwargs['model'] = Models.GEMMA3.value
        return ollama.chat(
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            **kwargs,
        )

    def get_stream(
        self, prompt: str, **kwargs
    ) -> c.Iterator[ollama.ChatResponse]:
        kwargs['stream'] = True
        return self.get_chat(prompt, **kwargs)

    def get_response(self, prompt: str, **kwargs):
        return self.get_chat(prompt, **kwargs)

    def get_base_prompt(self):
        context = f"""For the {self.county} county the prediction was {self.prediction:.3f}.
XAI methods were employed to explain the predictions which
provided attribution for features. As we are dealing with a regression task, the positive attributions
mean that the feature pushed the prediction up and the negative attributions mean that the feature
pushed the prediction down."""
        return f"""
Dataset description: {self.dataset_description}

Input data: {self.input_description}

Output data: {self.output_description}

Context: {context}
"""


def format_attributions(attributions: c.Sequence[ExplainerOutput]):
    prompt = ''
    for output in attributions:
        for attribution in output['attributions']:
            attribution_text = pprint.pformat(
                [asdict(feature) for feature in attribution],
                indent=1,
                sort_dicts=False,
            )
            prompt = '\n\n'.join([prompt, attribution_text])
    return prompt.strip()


@dataclass
class AttributionStory(BaseRequester):
    attributions: c.Sequence[ExplainerOutput]

    @property
    def story_prompt(self):
        story_context = """Can you come up with a plausible, fluent story as to why the model could have
predicted this outcome, based on the most influential positive and most influential
negative attribution values? The features were provided and are sorted by their attribution absolute value.
Try to explain the most important feature values in this story, as
well as potential interactions that fit the story. No need to enumerate individual
features outside of the story. Conclude with a short summary of why this
prediction may have occurred. It is very important that you limit your answer to 8 sentences.
Only write the story, without any other paragraphs. Do not talk about the features in abstract terms."""
        attribution_format = {
            name: field.type
            for name, field in self.attributions[0]['attributions'][0][
                0
            ].__dataclass_fields__.items()
        }
        return f"""
Attributions: {format_attributions(self.attributions)}

Attribution format: {attribution_format}

Story: {story_context}
"""

    def get_story_prompt(self) -> str:
        prompt = self.get_base_prompt()
        return '\n'.join([prompt, self.story_prompt])

    def get_story_stream(self):
        story_prompt = self.get_story_prompt()
        return (story_prompt, self.get_stream(story_prompt))

    def get_story(self) -> tuple[str, ollama.ChatResponse]:
        story_prompt = self.get_story_prompt()
        return (story_prompt, self.get_response(story_prompt))


class StoryEvaluatorResponse(pydantic.BaseModel):
    ideal_story_description: str = pydantic.Field(
        title='Ideal story',
        description='A description of what the ideal story should contain. This should include a detailed explanation.',
    )
    reason: str = pydantic.Field(
        title='Best story reason',
        description='The reason why the best story was chosen with respect to the provided properties. This should include a detailed explanation.',
    )
    story: int = pydantic.Field(
        title='Best story number',
        description='The number of the best story. This should be the number of the story described in the "reason" field.',
    )


@dataclass
class StoryEvaluator(BaseRequester):
    stories: c.Sequence[str]

    @property
    def evaluator_prompt(self) -> str:
        return """Stories were generated based on feature attributions using XAI methods.
Compare the stories using the context and the coherence properties. Analyze the stories and
return the one which better fits the properties.
Context - Describes how relevant the explanation is to the user and their needs.
Coherence - Describes how accordant the explanation is with prior knowledge and beliefs.

{stories}


Best story: Start by describing what a good story should contain with respect to the context and coherence properties.
After that, mention the reason on why you picked the story and the number of the story you picked."""

    def get_best_story_stream(self):
        prompt = self.get_best_story_prompt()
        kwargs = {'format': StoryEvaluatorResponse.model_json_schema()}
        return (prompt, self.get_stream(prompt, **kwargs))

    def get_best_story_prompt(self):
        base_prompt = self.get_base_prompt()
        story_prompt = ''
        for i, story in enumerate(self.stories, start=1):
            story_prompt = '\n\n'.join([story_prompt, f'Story {i}: {story}'])
        evaluator_prompt = self.evaluator_prompt.format(stories=story_prompt)
        return '\n'.join([base_prompt, evaluator_prompt])

    def get_best_story(self) -> StoryEvaluatorResponse:
        kwargs = {'format': StoryEvaluatorResponse.model_json_schema()}
        response = self.get_response(self.get_best_story_prompt(), **kwargs)
        parsed = StoryEvaluatorResponse.model_validate_json(
            response['message']['content']
        )
        return parsed


class Accuracy(pydantic.BaseModel):
    chain_of_thought: str
    assessment: int


class Completeness(pydantic.BaseModel):
    chain_of_thought: str
    assessment: int


@dataclass
class Grader(BaseRequester):
    story: str
    attributions: c.Sequence[ExplainerOutput]

    @property
    def accuracy_prompt(self) -> str:
        question = """How accurate is the information in the story, based on the attributions given? A story can score 1 even if it is missing
information as long as everything in the story is correct. If the story mentions features that are not provided as attributions, it scores 0.
Make sure the contribution direction is correct - positive contributions increase the output, negative contributions decrease the output."""
        rubric = 'Rubric: 0 - Contains one or more errors in value or contribution direction. 1 - Contains no errors, but may be missing information.'
        assessment = 'A single number from the options in the rubric. Provide only a single number with no other text'
        return f"""
Now, please assess a story based on a rubric.

Follow the following format.
Question: {question}
Story: {self.story}
Rubric: {rubric}
Assessment: {assessment}
Attributions: {format_attributions(self.attributions)}

Start by listing out all the features in the story, 
and then for each one compare it to the actual attributions to ensure its value
is approximately correct. Only write your chain of thought and your assessment, without
any other paragragh before or after.
    """

    @property
    def completeness_prompt(self) -> str:
        question = (
            'How completely does the narrative below describe the attributions?'
        )
        rubric = f"""
0 - Some features were not mentioned.
1 - First {min([5, len(self.attributions[0]['attributions'][0])])} features were mentioned."""
        assessment = 'A single number from the options in the rubric. Provide only a single number with no other text'
        return f"""
Now, please assess a story based on a rubric.

Follow the following format.
Question: {question}
Story: {self.story}
Rubric: {rubric}

Assessment: {assessment}

Attributions: {format_attributions(self.attributions)}

Start by listing out all the features and their attribution score in the Attributions field,
and then determine if all the features are present in the story."""

    def grade_completeness(self):
        base = self.get_base_prompt()
        prompt = '\n\n'.join((base, self.completeness_prompt))
        kwargs = {'format': Completeness.model_json_schema()}
        return (prompt, self.get_stream(prompt, **kwargs))

    def grade_accuracy(self):
        base = self.get_base_prompt()
        prompt = '\n\n'.join((base, self.accuracy_prompt))
        kwargs = {'format': Accuracy.model_json_schema()}
        return (prompt, self.get_stream(prompt, **kwargs))


def get_modis_bands_groups() -> tuple[list[int], dict[int, str]]:
    group_labels = {
        0: 'MODIS visible bands',
        1: 'MODIS NIR bands',
        2: 'MODIS SWIR bands',
        3: 'MODIS LST bands',
    }
    band_label_mapper = [0, 1, 0, 0, 1, 2, 2, 3, 3]
    return (band_label_mapper, group_labels)


def get_crop_calendar_groups(
    timestamps: list[str],
) -> tuple[list[int], dict[int, str]]:
    dummy_year = 1970
    soybean_calendar_labels = {
        0: 'Planting season',
        1: 'Mid season',
        2: 'Harvesting season',
    }
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
    modis_band_groups, modis_group_labels = get_modis_bands_groups()
    multivariate_timeseries_explainer = MultivariateTimeseriesExplainer(
        model,
        dataset._timestamps,
        dataset._feature_names,
    )
    multivariate_grouped_timeseries_explainer = MultivariateTimeseriesExplainer(
        model,
        list(crop_calendar_labels.values()),
        list(modis_group_labels.values()),
        crop_calendar_groups,
        modis_band_groups,
    )
    channel_integrated_gradients = channel_explainer.integrated_gradients(data)

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
    multivariate_grouped_feature_ablation = (
        multivariate_grouped_timeseries_explainer.feature_ablation(data)
    )
    multivariate_grouped_kernel_shap = (
        multivariate_grouped_timeseries_explainer.kernel_shap(data)
    )
    story = AttributionStory(
        dataset._dataset_description,
        dataset._input_description,
        dataset._output_description,
        county=str(dataset.locations[index]),
        # target=round(float(target.cpu()), 4),
        prediction=round(float(prediction.detach().cpu()), 4),
        attributions=[
            multivariate_grouped_kernel_shap,
        ],
    )
    prompt = story.get_prompt()
    stream = story.get_stream(prompt)
    content = ''
    for chunk in stream:
        if 'message' in chunk and 'content' in chunk['message']:
            content += chunk['message']['content']
        print(chunk['message']['content'], flush=True, end='')

    for prompt, stream in get_grades(
        content, story.attributions, index, prediction, dataset
    ):
        collected = ''
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                collected += chunk['message']['content']
            print(chunk['message']['content'], flush=True, end='')
