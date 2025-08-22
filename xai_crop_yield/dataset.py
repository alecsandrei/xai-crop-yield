from __future__ import annotations

import bisect
import collections.abc as c
import itertools
import operator
import typing as t
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchgeo.datasets import SustainBenchCropYield

from xai_crop_yield.config import DEVICE, RAW_DATA_DIR


class SustainBenchLocation(t.NamedTuple):
    state: str
    county: str

    @classmethod
    def from_string(cls: t.Type[t.Self], string: str) -> t.Self:
        split = string.split('_')
        assert len(split) == 3
        return cls(split[0], split[1])


@dataclass
class SustainBenchCropYieldTimeseries(Dataset):
    root: Path
    years: c.Sequence[int]
    country: str = 'usa'
    locations: c.Sequence[SustainBenchLocation] = field(init=False)
    data: c.Sequence[tuple[torch.Tensor, torch.Tensor]] = field(init=False)

    def _handle_split(
        self, split: str
    ) -> tuple[SustainBenchCropYield, c.Sequence[str]]:
        dataset = SustainBenchCropYield(
            self.root, download=True, countries=[self.country], split=split
        )
        data_dir = self.root / dataset.dir / self.country
        locations: np.ndarray = np.load(data_dir / f'{split}_keys.npz')['data']
        return (dataset, locations.tolist())

    def _load(self):
        splits = [
            self._handle_split(split) for split in ('train', 'test', 'dev')
        ]
        chain_dataset = itertools.chain.from_iterable(
            iter(split[0]) for split in splits
        )
        chain_locations = itertools.chain.from_iterable(
            split[1] for split in splits
        )
        data = defaultdict(list)  # type: ignore
        for dataset, location in zip(chain_dataset, chain_locations):
            dataset['year'] = int(dataset['year'])  # type: ignore
            if dataset['year'] not in self.years:
                continue
            list_ = data[SustainBenchLocation.from_string(location)]
            bisect.insort(list_, dataset, key=operator.itemgetter('year'))

        filtered_data = {
            location: yearly_data
            for location, yearly_data in data.items()
            if all(
                year in [data['year'] for data in yearly_data]
                for year in self.years
            )
        }

        def concatenate_location_data(
            location: SustainBenchLocation,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            location_data = filtered_data[location]
            images = torch.cat(
                [entry['image'].unsqueeze(0) for entry in location_data[:-1]]
            ).to(DEVICE)
            assert images.size(0) == len(location_data) - 1
            # targets = torch.Tensor(
            #    [entry['label'] for entry in location_data]
            # ).to(DEVICE)
            target = location_data[-1]['label'].to(DEVICE)
            return (images, target)

        self.locations = list(filtered_data)
        self.data = [
            concatenate_location_data(location) for location in filtered_data
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = SustainBenchCropYieldTimeseries(
        RAW_DATA_DIR, country='usa', years=list(range(2005, 2014))
    )
    dataset._load()
