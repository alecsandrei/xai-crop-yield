from __future__ import annotations

import numpy as np
import torch

from xai_crop_yield import (  # noqa: F401
    config,
    dataset,
    features,
    modeling,
    plots,
)

torch.manual_seed(0)
np.random.seed(config.SEED)
