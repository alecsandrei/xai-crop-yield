from __future__ import annotations

import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from tqdm import tqdm

from xai_crop_yield.config import (
    DEVICE,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
)
from xai_crop_yield.dataset import get_dataset
from xai_crop_yield.features import get_county_data
from xai_crop_yield.modeling.train import get_model
from xai_crop_yield.modeling.xai import (
    ChannelExplainer,
    FeatureAttribution,
    HeatmapExplainer,
    TimeseriesExplainer,
    get_crop_calendar_groups,
    get_modis_bands_groups,
)


def plot_heatmaps(index: int, show: bool = False):
    model = get_model()
    dataset = get_dataset()
    data, target = dataset[index]
    data = data.to(DEVICE).unsqueeze(0)
    target = target.to(DEVICE).view(-1, 1)
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), share_all=True)
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    explainer = HeatmapExplainer(model)
    deeplift_attributions = explainer.deeplift(data)
    occlusion_attributions = explainer.occlusion(data)
    assert isinstance(deeplift_attributions['attributions'], torch.Tensor)

    explainer.plot(
        deeplift_attributions['attributions'].sum(dim=[1, 2])[0], ax=grid[0]
    )
    grid[0].set_title(deeplift_attributions['method'])
    grid[0].set_xlabel('Temporal dimension')
    grid[0].set_ylabel('Histogram bins')
    assert isinstance(occlusion_attributions['attributions'], torch.Tensor)
    explainer.plot(occlusion_attributions['attributions'][0], ax=grid[1])
    grid[1].set_title(occlusion_attributions['method'])
    grid[1].set_xlabel('Temporal dimension')
    if show:
        plt.show()
    fig.savefig(
        FIGURES_DIR / f'heatmaps_{dataset.locations[index]}.png',
        dpi=300,
        bbox_inches='tight',
    )


def plot_bands(index: int, show: bool = False):
    dataset = get_dataset()
    images, _ = dataset[index]
    arr = images[-1].detach().cpu().numpy()  # year before prediction
    fig = plt.figure(figsize=(9, 9))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(3, 3),
        axes_pad=(0.2, 0.4),
        share_all=True,
        cbar_mode='single',
    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])
    clim = (np.min(images), np.max(images))
    for i, image in enumerate(arr):
        im = grid[i].imshow(image, clim=clim)
        grid[i].set_title(dataset._feature_names[i])
    grid[i].cax.colorbar(im)
    plt.tight_layout()
    if show:
        plt.show()
    fig.savefig(
        FIGURES_DIR / f'bands_{dataset.locations[index]}.png',
        dpi=300,
        bbox_inches='tight',
    )


def euclidean_distance(arr1: np.ndarray, arr2: np.ndarray) -> np.floating:
    return np.linalg.norm(arr1 - arr2)


def save_attribution_distance_data(index: int):
    model = get_model()
    dataset = get_dataset()
    counties = get_county_data()

    channel_attributions = defaultdict(dict)  # type:ignore
    channel_explainer = ChannelExplainer(model, dataset._feature_names)
    for i in tqdm(range(len(dataset))):
        data, target = dataset[i]
        data = data.to(DEVICE).unsqueeze(0)
        explanations = channel_explainer.feature_ablation(data)['attributions'][
            0
        ]
        for explanation in explanations:
            channel_attributions[i][explanation[0]] = explanation[1]
        channel_attributions[i]['NAME'] = dataset.locations[i].county.lower()
        channel_attributions[i]['STUSPS'] = dataset.locations[i].state.upper()
        channel_attributions[i]['index'] = i
    attributions = pd.DataFrame.from_dict(channel_attributions, orient='index')
    attribution = attributions.loc[index, dataset._feature_names]
    attributions['distance'] = attributions[dataset._feature_names].apply(
        euclidean_distance, arr2=attribution, axis='columns'
    )
    distance_gdf = counties.merge(
        attributions,
        right_on=['STUSPS', 'NAME'],
        left_on=['STUSPS', 'NAME_LOWERCASE'],
        suffixes=('', '_y'),
    )
    distance_gdf.to_file(PROCESSED_DATA_DIR / 'similarities.fgb')


def plot_attributions(
    attributions: FeatureAttribution,
    show: bool = False,
):
    if isinstance(attributions, torch.Tensor):
        raise NotImplementedError()
    df = pd.DataFrame(attributions, columns=['name', 'attribution'])
    max = df['attribution'].abs().max() * 1.2

    fig, ax = plt.subplots()
    ax.bar(
        df['name'],
        df['attribution'],
        color=np.where(df['attribution'] < 0, 'crimson', 'blue'),
    )
    ax.set_ylim(-max, max)
    ax.set_ylabel('Attribution')
    ax.axhline(0, color='black')
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_channel_attributions(index: int):
    directory = FIGURES_DIR / 'channel_attributions'
    directory.mkdir(exist_ok=True)
    model = get_model()
    dataset = get_dataset()
    location = dataset.locations[index]
    data, target = dataset[index]
    data = data.to(DEVICE).unsqueeze(0)
    groups, labels = get_modis_bands_groups(dataset._timestamps)
    grouped_channel_explainer = ChannelExplainer(
        model, list(labels.values()), groups, sort=False
    )
    grouped_channel_attributions = grouped_channel_explainer.feature_ablation(
        data
    )['attributions'][0]
    fig = plot_attributions(grouped_channel_attributions, show=False)
    plt.title(f'MODIS band attribution for {location}')
    plt.xlabel('Bands')
    fig.savefig(
        directory
        / f'{dataset.locations[index]}_grouped_channels_attributions.png',
        bbox_inches='tight',
        dpi=300,
    )
    plt.close(fig)

    channel_explainer = ChannelExplainer(
        model, dataset._feature_names, sort=False
    )
    channel_attributions = channel_explainer.feature_ablation(data)[
        'attributions'
    ][0]
    fig = plot_attributions(channel_attributions, show=False)
    plt.xlabel('Bands')
    plt.xticks(fontsize=9)
    plt.title(f'MODIS band attribution for {location}')

    fig.savefig(
        directory / f'{dataset.locations[index]}_channel_attributions.png',
        bbox_inches='tight',
        dpi=300,
    )
    plt.close(fig)


def plot_timeseries_attributions(index: int):
    directory = FIGURES_DIR / 'timeseries_attributions'
    directory.mkdir(exist_ok=True)
    model = get_model()
    dataset = get_dataset()
    location = dataset.locations[index]
    data, target = dataset[index]
    data = data.to(DEVICE).unsqueeze(0)
    groups, crop_calendar_labels = get_crop_calendar_groups(dataset._timestamps)
    crop_calendar_explainer = TimeseriesExplainer(
        model, list(crop_calendar_labels.values()), groups, sort=False
    )
    crop_calendar_attributions = crop_calendar_explainer.feature_ablation(data)[
        'attributions'
    ][0]
    fig = plot_attributions(crop_calendar_attributions, show=False)
    plt.title(f'Crop calendar attribution for {location}')
    plt.xlabel('Crop calendar')
    fig.savefig(
        directory
        / f'{dataset.locations[index]}_crop_calendar_attributions.png',
        bbox_inches='tight',
        dpi=300,
    )
    plt.close(fig)

    timestamps = [
        datetime.datetime.strptime(timestamp, '%B-%d').strftime('%d/%m')
        for timestamp in dataset._timestamps
    ]
    timeseries_explainer = TimeseriesExplainer(model, timestamps, sort=False)
    timeseries_attributions = timeseries_explainer.feature_ablation(data)[
        'attributions'
    ][0]
    fig = plot_attributions(timeseries_attributions, show=False)
    plt.xticks(rotation=90, fontsize=10)
    plt.xlabel('Date')
    plt.title(f'Timeseries attribution for {location}')

    fig.savefig(
        directory / f'{dataset.locations[index]}_timeseries_attributions.png',
        bbox_inches='tight',
        dpi=300,
    )
    plt.close(fig)


if __name__ == '__main__':
    dataset = get_dataset()
    for index in range(len(dataset)):
        plot_channel_attributions(index)
        plot_timeseries_attributions(index)
    # index = 50
    # plot_attributions(index, show=True)
    # breakpoint()

    # save_attribution_distance_data(index)
    # plot_bands(index)
    # plot_heatmaps(index)
