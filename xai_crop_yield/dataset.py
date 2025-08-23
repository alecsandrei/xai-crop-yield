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
from torch.utils.data import DataLoader, Dataset, random_split
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

    def __str__(self) -> str:
        return f'{self.county}, {self.state}'


@dataclass
class SustainBenchCropYieldTimeseries(Dataset):
    """Descriptions were taken from https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/crop_yield.html"""

    root: Path
    years: c.Sequence[int]
    country: str = 'usa'
    locations: c.Sequence[SustainBenchLocation] = field(init=False)
    data: c.Sequence[tuple[torch.Tensor, torch.Tensor]] = field(init=False)

    @property
    def _dataset_description(self) -> str:
        return (
            'County-level yields from the USA were chosen for the analysis. '
            'The inputs are spectral band and temperature histograms over each county '
            'for the harvest season, derived from MODIS satellite images of each region. '
            'The outputs are soybean yields in metric tonnes per harvested hectare over the counties.'
        )

    @property
    def _input_description(self) -> str:
        return (
            'The input is a 32x32x9 band histogram over a county’s harvest season. '
            'For each of 7 surface reflectance and 2 surface temperature bands '
            'we bin MODIS pixel values into 32 ranges and 32 timesteps per harvest season.'
        )

    @property
    def _output_description(self) -> str:
        return 'The output is the soybean yield over the harvest season, in metric tonnes per harvested hectare.'

    @property
    def _feature_names(self) -> list[str]:
        return [
            'b1_RED_620-670nm',
            'b2_NIR_841-876nm',
            'b3_BLUE_459-479nm',
            'b4_GREEN_545-565nm',
            'b5_NIR_1230-1250nm',
            'b6_SWIR_1628-1652nm',
            'b7_SWIR_2105-2155nm',
            'daytime_LST',
            'nighttime_LST',
        ]

    def _get_readable_location(self, index: int):
        return self.locations[index]

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
            target = location_data[-1]['label'].to(DEVICE)
            return (images, target)

        self.locations = list(filtered_data)
        self.data = [
            concatenate_location_data(location) for location in filtered_data
        ]

    def _get_dataloaders(
        self,
        train_test_validation_split: tuple[float, float, float],
        batch_size: int = 32,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        np.testing.assert_almost_equal(sum(train_test_validation_split), 1)
        train, test, val = random_split(self, train_test_validation_split)
        train_dataloader = DataLoader(
            train, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)
        return (train_dataloader, test_dataloader, val_dataloader)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = SustainBenchCropYieldTimeseries(
        RAW_DATA_DIR, country='usa', years=list(range(2005, 2014))
    )
    dataset._load()
