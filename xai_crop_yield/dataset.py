from __future__ import annotations

import bisect
import collections.abc as c
import itertools
import operator
import typing as t
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchgeo.datasets import SustainBenchCropYield

from xai_crop_yield.config import DEVICE, RAW_DATA_DIR


class SustainBenchLocation(t.NamedTuple):
    county: str
    state: str

    @classmethod
    def from_string(cls: t.Type[t.Self], string: str) -> t.Self:
        split = string.split('_')
        assert len(split) == 3
        return cls(split[0], split[1])

    def __str__(self) -> str:
        return f'{self.county.title()}, {self.state.upper()}'


@dataclass
class SustainBenchCropYieldTimeseries(Dataset):
    """Descriptions were taken from https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/crop_yield.html"""

    root: Path
    years: c.Sequence[int]
    country: str = 'usa'
    locations: c.Sequence[SustainBenchLocation] = field(init=False)
    data: c.Sequence[tuple[torch.Tensor, torch.Tensor]] = field(init=False)

    def __post_init__(self):
        self._load()

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
            'Red 620-670nm',
            'NIR 841-876nm',
            'Blue 459-479nm',
            'Green 545-565nm',
            'NIR 1230-1250nm',
            'SWIR 1628-1652nm',
            'SWIR 2105-2155nm',
            'Daytime LST',
            'Nighttime LST',
        ]

    @property
    def _timestamps(self) -> list[str]:
        delta = timedelta(days=8)
        dateformat = '%B-%d'

        dates = []
        curr_date = datetime(year=2000, month=2, day=26)  # year is arbitrary
        for _ in range(32):
            dates.append(curr_date.strftime(dateformat))
            curr_date += delta
        assert len(dates) == 32
        return dates

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
                [entry['image'].unsqueeze(0) for entry in location_data]
            ).to(DEVICE)
            targets = torch.cat(
                [entry['label'].unsqueeze(0) for entry in location_data]
            ).to(DEVICE)
            return (images, targets)

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
        images, targets = self.data[index]
        images = images[:-1]
        target = targets[-1]
        return (images, target)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = SustainBenchCropYieldTimeseries(
        RAW_DATA_DIR, country='usa', years=list(range(2005, 2014))
    )
