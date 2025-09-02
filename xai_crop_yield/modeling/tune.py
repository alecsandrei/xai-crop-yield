from __future__ import annotations

import typing as t
from collections import Counter
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning import Callback, Trainer
from loguru import logger
from ray import train, tune
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from xai_crop_yield.modeling.xai import ExplainerOutput

if t.TYPE_CHECKING:
    from xai_crop_yield.modeling.train import ConvLSTMModel
    from xai_crop_yield.modeling.xai import (
        Explainer,
        Features,
    )


class TuneSpace(t.TypedDict):
    hidden_dim: int
    learning_rate: float


@dataclass
class Hyperparameters:
    hidden_dim: int
    learning_rate: float

    @staticmethod
    def get_tune_space() -> TuneSpace:
        return {
            'hidden_dim': tune.choice([1, 5, 10, 20]),
            'learning_rate': tune.loguniform(1e-3, 1e-1),
        }


def add_gaussian_noise(X, sigma=0.01):
    """
    Add small Gaussian noise to input X and clamp to [0,1].
    sigma: standard deviation of noise
    """
    noise = torch.randn_like(X).to(X.device) * sigma
    X_noisy = X + noise
    X_noisy = X_noisy.clamp(0.0, 1.0)  # keep histogram values valid
    return X_noisy


@dataclass
class Continuity:
    model: ConvLSTMModel
    attribution_explainer: Explainer
    values: list[float] = field(default_factory=list)

    @staticmethod
    def attributions_to_ndarray(attributions: ExplainerOutput) -> np.ndarray:
        features_attributions: list[list[float]] = []
        for features in attributions['attributions']:
            feature_attributions = []
            for feature in features:
                feature_attributions.append(feature.attribution)
            features_attributions.append(feature_attributions)

        arr = np.array(features_attributions)
        assert arr.ndim == 2

        return arr

    def update(self, x: torch.Tensor) -> float:
        X_noise = add_gaussian_noise(x, sigma=0.05)
        if hasattr(self.attribution_explainer, 'sort'):
            if self.attribution_explainer.sort:
                raise ValueError(
                    f'Sort has to be none in {self.attribution_explainer}'
                )
        expl_X = self.attributions_to_ndarray(
            self.attribution_explainer.feature_ablation(x)
        )
        expl_X_noise = self.attributions_to_ndarray(
            self.attribution_explainer.feature_ablation(X_noise)
        )
        continuity = np.sqrt(((expl_X - expl_X_noise) ** 2).sum(1)).mean()
        self.values.append(continuity)
        return continuity

    def compute(self) -> float:
        return np.mean(self.values)

    def reset(self):
        self.values = []


@dataclass
class Shannon:
    model: ConvLSTMModel
    attribution_explainer: Explainer
    top_k: int = 5
    counter: Counter[str] = field(default_factory=Counter)

    def get_top_k_attributions(self, images: torch.Tensor) -> list[str]:
        attributions = self.attribution_explainer.feature_ablation(images)
        features_list: list[Features] = attributions['attributions']
        for features in features_list:
            features.sort_by_attribution(reverse=True)
        top_k = [features[: self.top_k] for features in features_list]
        return [feature.name() for feature in chain.from_iterable(top_k)]

    def shannon(self, probs: np.ndarray):
        return (-probs * np.log(probs)).sum()

    def update(self, x: torch.Tensor):
        features = self.get_top_k_attributions(x)
        self.counter.update(features)

    def compute(self):
        total = self.counter.total()
        probs = [val / total for val in self.counter.values()]
        shannon = self.shannon(np.array(probs))
        normalized_shannon = shannon / np.log(total)
        return normalized_shannon

    def reset(self):
        self.counter.clear()


def get_scheduler(metric: str, mode: str, epochs: int):
    return ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=epochs,
        grace_period=min(epochs, 50),
        reduction_factor=2,
    )


def get_tuner(
    train_func,
    func_kwds: dict,
    metric: str,
    mode: str,
    epochs: int,
    num_trials: int,
    trial_dir: Path,
    checkpoint_kwargs: dict[str, t.Any] | None = None,
    run_config_kwargs: dict[str, t.Any] | None = None,
) -> tune.Tuner:
    def get_run_config():
        return tune.RunConfig(
            storage_path=trial_dir,
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=50,
                checkpoint_score_attribute=metric,
                checkpoint_score_order=mode,
                **checkpoint_kwargs if checkpoint_kwargs is not None else {},
            ),
            **run_config_kwargs if run_config_kwargs is not None else {},
        )

    def get_tune_config():
        logger.warning('Schedulers from ray tune are disabled')
        return tune.TuneConfig(
            # scheduler=get_scheduler(metric, mode, epochs),
            num_samples=num_trials,
            search_alg=HyperOptSearch(metric=metric, mode=mode),
        )

    def get_trainable():
        return TorchTrainer(
            tune.with_parameters(train_func, **func_kwds),
            scaling_config=train.ScalingConfig(use_gpu=True),
            run_config=get_run_config(),
        )

    def get_param_space():
        return {
            'train_loop_config': Hyperparameters.get_tune_space(),
        }

    if tune.Tuner.can_restore(trial_dir):
        # ResumeConfig is kind of buggy, should be used with caution
        # It also has wrong type hints, so we need to ignore them
        resume_config = tune.ResumeConfig(
            finished=tune.ResumeConfig.ResumeType.RESUME,  # type: ignore
            unfinished=tune.ResumeConfig.ResumeType.RESUME,  # type: ignore
            errored=tune.ResumeConfig.ResumeType.RESUME,  # type: ignore
        )
        restored = tune.Tuner.restore(
            trial_dir.as_posix(),
            trainable=get_trainable(),  # type: ignore
            param_space=get_param_space(),
            _resume_config=resume_config,
        )
        return restored
    else:
        restored = tune.Tuner(
            get_trainable(),
            param_space=get_param_space(),
            tune_config=get_tune_config(),
        )
    return restored


class StopAtLossThreshold(Callback):
    def __init__(self, monitor: str, min_value: float):
        super().__init__()
        self.monitor = monitor
        self.min_value = min_value

    def on_validation_end(
        self, trainer: Trainer, pl_module: pl.LightningModule
    ) -> None:
        try:
            metric = trainer.logged_metrics[self.monitor]
        except KeyError:
            return None
        should_stop = metric >= self.min_value
        if should_stop:
            should_stop = trainer.strategy.reduce_boolean_decision(
                True, all=False
            )
            trainer.should_stop = True


def get_trainer(epochs: int):
    stop_at_loss = StopAtLossThreshold('val_r2', min_value=0.7)
    return pl.Trainer(
        devices='auto',
        accelerator='gpu',
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback(), stop_at_loss],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
        max_epochs=epochs,
        # profiler='simple',
    )
