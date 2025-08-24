from __future__ import annotations

import pathlib

import altair as alt
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from xai_crop_yield.config import DEVICE, MODELS_DIR, RAW_DATA_DIR
from xai_crop_yield.dataset import SustainBenchCropYieldTimeseries
from xai_crop_yield.modeling.train import ConvLSTMModel
from xai_crop_yield.modeling.xai import (
    AttributionStory,
    ChannelExplainer,
    TimeseriesExplainer,
)

st.set_page_config(layout='wide')


STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
DOWNLOADS_PATH = STREAMLIT_STATIC_PATH / 'downloads'
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()

YEARS = list(range(2005, 2016))
YEARS_STR = [str(year) for year in YEARS]

ATTRIBUTION_METHODS = ['Feature ablation', 'Kernel SHAP']


@st.cache_data
def get_geom_data():
    links = {
        'state': RAW_DATA_DIR / 'us_states.geojson',
        'county': RAW_DATA_DIR / 'us_counties.geojson',
    }
    state = gpd.read_file(links['state'])[['STUSPS', 'STATEFP', 'NAME']]
    state.rename(columns={'NAME': 'STATE_NAME'}, inplace=True)
    county = gpd.read_file(links['county'])
    county = county.merge(state, on='STATEFP')
    county['NAME_LOWERCASE'] = county['NAME'].str.lower()
    yield_data = get_counties_yield_data()
    yield_data['NAME_LOWERCASE'] = yield_data['NAME'].str.lower()

    geom_data = gpd.GeoDataFrame(
        yield_data.merge(
            county, on=['STUSPS', 'NAME_LOWERCASE'], suffixes=('x_', '')
        ),
        geometry='geometry',
        crs=county.crs,
    )
    return geom_data


def get_county_geom_data(state: str, county: str):
    data = get_geom_data()
    masked = data[(data['STATE_NAME'] == state) & (data['NAME'] == county)]
    assert masked.shape[0] == 1
    return masked


def get_county_yield_data(state: str, county: str):
    data = get_county_geom_data(state, county)
    pivoted = data[YEARS_STR].T
    predicted = data[f'predicted_{YEARS[-1]}'].iloc[0]
    pivoted.loc[YEARS_STR[-1], 'predicted'] = predicted
    pivoted.reset_index(inplace=True)
    pivoted.columns = ['years', 'crop_yield', 'prediction']
    return (pivoted, predicted)


@st.cache_data
def get_states() -> list[str]:
    data = get_geom_data()
    return sorted(data['STATE_NAME'].dropna().unique())


@st.cache_data
def get_counties(state: str | None = None) -> list[str]:
    data = get_geom_data()
    mask = data['STATE_NAME'] == state
    return sorted(data.loc[mask, 'NAME'].dropna().unique())


@st.cache_data
def get_dataset():
    return SustainBenchCropYieldTimeseries(
        RAW_DATA_DIR, country='usa', years=YEARS
    )


@st.cache_data
def get_model():
    return ConvLSTMModel.load_from_checkpoint(
        MODELS_DIR / 'checkpoint.ckpt'
    ).to(DEVICE)


@st.cache_data
def get_counties_yield_data():
    dataset = get_dataset()
    model = get_model()
    counties_data = []
    for (images, targets), location in zip(dataset.data, dataset.locations):
        county_data = {}
        county_data['NAME'] = location.county.title()
        county_data['STUSPS'] = location.state.upper()
        for i, year in enumerate(YEARS_STR):
            county_data[year] = float(targets[i])
        output = float(
            model(images[:-1].unsqueeze(0).to(DEVICE)).detach().cpu()
        )
        county_data[f'predicted_{YEARS[-1]}'] = output

        counties_data.append(county_data)
    df = pd.DataFrame(counties_data)
    df['residuals'] = df[YEARS_STR[-1]] - df[f'predicted_{YEARS[-1]}']
    return df


def get_crop_yield_chart(state: str, county: str):
    yield_data, predicted = get_county_yield_data(state, county)
    numeric_cols = ['crop_yield', 'prediction']
    base = alt.Chart(yield_data)
    line = (
        base.mark_line()
        .encode(
            x='years',
            y=alt.Y(
                'crop_yield',
                scale=alt.Scale(
                    domain=[
                        np.nanmin(yield_data[numeric_cols].values) * 0.8,
                        np.nanmax(yield_data[numeric_cols].values) * 1.2,
                    ]
                ),
                title='crop yield',
            ),
        )
        .interactive()
    )
    point = (
        base.mark_point(color='red', filled=True, size=120)
        .encode(
            x='years',
            y=alt.Y(
                'prediction',
                title='crop yield',
            ),
        )
        .interactive()
    )
    text = base.mark_text(dy=-10, align='center', color='white').encode(
        x='years', y='crop_yield', text=alt.Text('crop_yield:Q', format=',.2f')
    )
    chart = line + point + text
    return chart


def get_attribution_chart(output, is_timeseries: bool = False):
    data = pd.DataFrame(output['attributions'][0], columns=['x', 'y'])
    if is_timeseries:
        data['x'] = pd.to_datetime(data['x'], format='%B-%d')
    base = alt.Chart(data)
    max = data['y'].abs().max() * 1.2
    y = alt.Y(
        'y:Q',
        scale=alt.Scale(domain=[-max, max]),
        title='attribution',
    )
    x = alt.X(
        'x',
        title='',
        axis=alt.Axis(labelAngle=90),
    )

    bar = (
        base.mark_bar()
        .encode(
            x=x,
            y=y,
        )
        .interactive()
    )
    text = base.encode(x='x', y='y', text=alt.Text('y:Q', format=',.2f'))
    text_above = text.transform_filter(alt.datum.y > 0).mark_text(
        align='center', baseline='middle', fontSize=11, dy=-10, color='white'
    )

    text_below = text.transform_filter(alt.datum.y < 0).mark_text(
        align='center', baseline='middle', fontSize=11, dy=12, color='white'
    )
    return bar + text_above + text_below


def get_attribution(index: int, method: str):
    dataset = get_dataset()
    model = get_model()
    method_name = method.lower().replace(' ', '_')
    location = dataset.locations[index]
    images, target = dataset[index]
    images = images.to(DEVICE).unsqueeze(0)
    target = float(target.cpu())
    prediction = float(model(images).detach().cpu())
    channel_explainer = ChannelExplainer(model, dataset._feature_names)
    channel_attribution = getattr(channel_explainer, method_name)(images)
    timeseries_explainer = TimeseriesExplainer(model, dataset._timestamps)
    timeseries_attribution = getattr(timeseries_explainer, method_name)(images)
    # multivariate_timeseries_explainer = MultivariateTimeseriesExplainer(
    #    model, dataset._feature_names, dataset._timestamps
    # )
    # multivariate_timeseries_attribution = getattr(
    #    multivariate_timeseries_explainer, method_name
    # )(images)
    story = AttributionStory(
        dataset._dataset_description,
        dataset._input_description,
        dataset._output_description,
        str(location),
        prediction,
        target,
        channel_explainers=[channel_attribution],
        timeseries_explainers=[timeseries_attribution],
        # multivariate_timeseries_explainers=[
        #    multivariate_timeseries_attribution
        # ],
    )
    prompt = story.build()

    return (
        prompt,
        story.get_stream(prompt),
        get_attribution_chart(channel_attribution),
        get_attribution_chart(timeseries_attribution, is_timeseries=True),
    )


def get_colors_from_values(values):
    max_value = max(abs(values))
    norm = plt.Normalize(vmin=-max_value, vmax=max_value)
    cmap = plt.cm.seismic  # type: ignore
    return [cmap(norm(v)) for v in values]


def add_colors_to_data(gdf: gpd.GeoDataFrame):
    sorted = gdf.sort_values('residuals')
    colors = get_colors_from_values(sorted['residuals'])
    for index, color in zip(sorted.index, colors):
        r, g, b, _ = color
        gdf.loc[index, 'R'] = r * 255
        gdf.loc[index, 'G'] = g * 255
        gdf.loc[index, 'B'] = b * 255


def app():
    gdf = get_geom_data()
    gdf['residuals_fmt'] = gdf['residuals'].round(3)
    add_colors_to_data(gdf)
    row1_col1, row1_col2 = st.columns([0.5, 0.5])
    with row1_col1:
        state = st.selectbox('State', get_states())

    with row1_col2:
        county = st.selectbox('County', get_counties(state))
    county_centroid = (
        get_county_geom_data(state, county)['geometry'].iloc[0].centroid
    )

    initial_view_state = pdk.ViewState(
        latitude=county_centroid.y,
        longitude=county_centroid.x,
        zoom=3,
        max_zoom=16,
        pitch=0,
        bearing=0,
        height=900,
        width=None,
    )

    color_exp = '[R, G, B]'

    geojson = pdk.Layer(
        'GeoJsonLayer',
        gdf,
        pickable=True,
        opacity=0.5,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=True,
        get_elevation='residuals',
        elevation_scale=1,
        get_fill_color=color_exp,
        get_line_color=[0, 0, 0],
        get_line_width=2,
        line_width_min_pixels=1,
    )
    layers = [geojson]

    tooltip_html = '<br>'.join(
        [
            '<b>County:</b> {NAME}',
            '<b>State:</b> {STATE_NAME}',
            '<b>Prediction error:</b> {residuals_fmt}',
        ]
    )

    tooltip = {
        'html': tooltip_html,
        'style': {'backgroundColor': 'steelblue', 'color': 'white'},
    }
    r = pdk.Deck(
        layers=layers,
        initial_view_state=initial_view_state,
        map_style='light',
        tooltip=tooltip,
    )

    row2_col1 = st.columns(1)[0]
    with row2_col1:
        st.pydeck_chart(r)

    row3_col1 = st.columns(1)[0]

    with row3_col1:
        st.altair_chart(get_crop_yield_chart(state, county))

    row4_col1, row4_col2 = st.columns([0.5, 0.5])
    row5_col1, row5_col2 = st.columns([0.3, 0.7])
    row6_col1 = st.columns(1)[0]

    def write_explanation():
        index = gdf[
            (gdf['STATE_NAME'] == state) & (gdf['NAME'] == county)
        ].index[0]
        prompt, stream, channel_chart, timeseries_chart = get_attribution(
            index, attribution_method
        )
        with row5_col1:
            st.altair_chart(channel_chart)
        with row5_col2:
            st.altair_chart(timeseries_chart)
        with row6_col1:
            st.chat_message('user').write(prompt)
            st.chat_message('assistant').write_stream(stream_handler(stream))

    with row4_col2:
        attribution_method = st.selectbox(
            'Attribution method', ATTRIBUTION_METHODS
        )
    with row4_col1:
        st.button('Generate explanations', on_click=write_explanation)


def stream_handler(stream):
    for chunk in stream:
        yield chunk['message']['content']


app()
