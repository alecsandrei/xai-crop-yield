from __future__ import annotations

from torch.utils.data import DataLoader

from xai_crop_yield.config import DEVICE, MODELS_DIR
from xai_crop_yield.dataset import get_dataset
from xai_crop_yield.modeling.train import ConvLSTMModel
from xai_crop_yield.modeling.tune import Continuity, Shannon
from xai_crop_yield.modeling.xai import ChannelExplainer


def test_shannon():
    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=8)
    model = ConvLSTMModel.load_from_checkpoint(
        MODELS_DIR / 'checkpoint.ckpt'
    ).to(DEVICE)
    shannon = Shannon(
        model,
        ChannelExplainer(
            model, channel_names=dataset._feature_names, sort=False
        ),
    )
    for images, target in dataloader:
        shannon.update(images)
        break
    shannon.compute()


def test_continuity():
    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=8)
    model = ConvLSTMModel.load_from_checkpoint(
        MODELS_DIR / 'checkpoint.ckpt'
    ).to(DEVICE)
    continuity = Continuity(
        model,
        ChannelExplainer(
            model, channel_names=dataset._feature_names, sort=False
        ),
    )
    for images, target in dataloader:
        continuity.update(images)
        break
    continuity.compute()
