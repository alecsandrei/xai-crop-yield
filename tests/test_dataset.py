from __future__ import annotations

from xai_crop_yield.config import RAW_DATA_DIR
from xai_crop_yield.dataset import SustainBenchCropYieldTimeseries


def test_sustain_bench_dataset():
    dataset = SustainBenchCropYieldTimeseries(
        RAW_DATA_DIR, country='usa', years=[2010, 2011]
    )
    dataset._load()
    assert hasattr(dataset, 'data')
    assert hasattr(dataset, 'locations')
