from __future__ import annotations

import collections.abc as c
import pathlib
from concurrent.futures import ThreadPoolExecutor
from threading import current_thread

import altair as alt
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import torch
from streamlit.runtime.scriptrunner import (
    add_script_run_ctx,
    get_script_run_ctx,
)
from torch import nn

import xai_crop_yield.dataset
import xai_crop_yield.modeling.train
from xai_crop_yield.config import DEVICE, YEARS
from xai_crop_yield.dataset import SustainBenchCropYieldTimeseries
from xai_crop_yield.features import get_county_data
from xai_crop_yield.modeling.xai import (
    AttributionStory,
    ChannelExplainer,
    ExplainerOutput,
    Features,
    Grader,
    MultivariateTimeseriesExplainer,
    StoryEvaluator,
    TimeseriesExplainer,
    get_crop_calendar_groups,
    get_modis_bands_groups,
)

st.set_page_config(layout='wide')


STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
DOWNLOADS_PATH = STREAMLIT_STATIC_PATH / 'downloads'
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()

YEARS_STR = [str(year) for year in YEARS]

ATTRIBUTION_METHODS = [
    'Kernel SHAP',
    'Feature ablation',
    'Integrated Gradients',
]

type MethodName = str
type AttributionFunc = c.Callable[[torch.Tensor, MethodName], ExplainerOutput]


@st.cache_data
def get_geom_data():
    county = get_county_data()
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
def get_dataset() -> SustainBenchCropYieldTimeseries:
    return xai_crop_yield.dataset.get_dataset()


@st.cache_data
def get_model() -> nn.Module:
    return xai_crop_yield.modeling.train.get_model()


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


def get_multivariate_chart(attributions: c.Sequence[ExplainerOutput]):
    methods = set(attribution['method'] for attribution in attributions)
    assert len(methods) == len(attributions), 'Expected different methods'
    dfs = []
    for output in attributions:
        assert len(output['attributions']) == 1, (
            'Only single batch is supported'
        )
        df = output['attributions'][0].as_df()
        df['method'] = output['method']
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    df['zero'] = 0
    base = alt.Chart(df, width=280)
    chart_attributions = base.mark_bar().encode(
        alt.Y('timestamp:N', sort=None).axis(None),
        # .title('method:N' if len(methods) > 1 else None),
        alt.X('attribution:Q', scale=alt.Scale(nice=True))
        .title(None)
        .axis(format=',.2f'),
        alt.Color('timestamp:N', sort=None)
        .title('Crop calendar')
        .legend(orient='right', titleOrient='top'),
    )
    chart_zero = base.mark_rule(color='red', width=3).encode(x='zero:Q')
    chart = (chart_attributions + chart_zero).facet(
        row=alt.Row('channel:N')
        .title(None)
        .header(labelAngle=0, labelAlign='left'),
        column=alt.Column('method:N', sort=None).title(None),
    )
    return chart


def get_channel_chart(features: Features):
    data = features.as_df()
    base = alt.Chart(data)
    max = data['attribution'].abs().max() * 1.2
    y = alt.Y(
        'attribution:Q',
        scale=alt.Scale(domain=[-max, max]),
        title='attribution',
    )
    x = alt.X('channel', title='', axis=alt.Axis(labelAngle=90))

    bar = base.mark_bar().encode(x=x, y=y).interactive()
    text = base.encode(
        x='channel:N',
        y='attribution:Q',
        text=alt.Text('attribution:Q', format=',.2f'),
    )
    text_above = text.transform_filter(alt.datum.attribution > 0).mark_text(
        align='center', baseline='middle', fontSize=11, dy=-10, color='white'
    )

    text_below = text.transform_filter(alt.datum.attribution < 0).mark_text(
        align='center', baseline='middle', fontSize=11, dy=12, color='white'
    )
    return bar + text_above + text_below


def get_timeseries_chart(features: Features):
    data = features.as_df()
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%B-%d')
    base = alt.Chart(data)
    max = data['attribution'].abs().max() * 1.2
    y = alt.Y(
        'attribution:Q',
        scale=alt.Scale(domain=[-max, max]),
        title='attribution',
    )
    x = alt.X('timestamp', title='', axis=alt.Axis(labelAngle=90))

    bar = base.mark_bar().encode(x=x, y=y).interactive()
    text = base.encode(
        x='timestamp',
        y='attribution',
        text=alt.Text('attribution:Q', format=',.2f'),
    )
    text_above = text.transform_filter(alt.datum.attribution > 0).mark_text(
        align='center', baseline='middle', fontSize=11, dy=-10, color='white'
    )

    text_below = text.transform_filter(alt.datum.attribution < 0).mark_text(
        align='center', baseline='middle', fontSize=11, dy=12, color='white'
    )
    return bar + text_above + text_below


def get_channel_attributions(input: torch.Tensor, method: str):
    channel_explainer = ChannelExplainer(
        get_model(), get_dataset()._feature_names
    )
    return getattr(channel_explainer, method)(input)


def get_timeseries_attributions(input: torch.Tensor, method: str):
    timeseries_explainer = TimeseriesExplainer(
        get_model(), get_dataset()._timestamps
    )
    return getattr(timeseries_explainer, method)(input)


def get_multivariate_attributions(
    input: torch.Tensor, method: str, flatten: bool = False
):
    dataset = get_dataset()
    modis_groups, modis_labels = get_modis_bands_groups()
    timestamp_groups, timestamp_labels = get_crop_calendar_groups(
        dataset._timestamps
    )
    multivariate_explainer = MultivariateTimeseriesExplainer(
        get_model(),
        # timestamps=dataset._timestamps,
        timestamps=list(timestamp_labels.values()),
        channel_names=list(modis_labels.values()),
        timestamp_groups=timestamp_groups,
        channel_groups=modis_groups,
    )

    return getattr(multivariate_explainer, method)(input)


@st.cache_data
def get_prediction(index: int) -> tuple[float, float]:
    dataset = get_dataset()
    model = get_model()
    images, target = dataset[index]
    images = images.to(DEVICE).unsqueeze(0)
    target = float(target.cpu())
    prediction = float(model(images).detach().cpu())
    return (target, prediction)


def get_attributions(
    index: int,
    methods: c.Sequence[str],
    attribution_funcs: c.Sequence[AttributionFunc],
) -> list[ExplainerOutput]:
    dataset = get_dataset()
    images, _ = dataset[index]
    images = images.to(DEVICE).unsqueeze(0)
    attributions = []
    for method in methods:
        method_func_name = method.lower().replace(' ', '_')
        for func in attribution_funcs:
            output = func(images, method_func_name)
            attributions.append(output)
    return attributions


def get_grades(
    story: str, attributions: c.Sequence[ExplainerOutput], index: int
):
    dataset = get_dataset()
    location = dataset.locations[index]
    _, prediction = get_prediction(index)
    grader = Grader(
        dataset._dataset_description,
        dataset._input_description,
        dataset._output_description,
        str(location),
        prediction,
        story=story,
        attributions=attributions,
    )
    return (grader.grade_accuracy(), grader.grade_completeness())


def get_comparison(stories: c.Sequence[str], index: int):
    dataset = get_dataset()
    location = dataset.locations[index]
    _, prediction = get_prediction(index)
    story = StoryEvaluator(
        dataset._dataset_description,
        dataset._input_description,
        dataset._output_description,
        str(location),
        prediction,
        stories=stories,
    )
    return story.get_best_story_stream()


def get_story(attributions: c.Sequence[ExplainerOutput], index: int):
    dataset = get_dataset()
    location = dataset.locations[index]
    _, prediction = get_prediction(index)
    story = AttributionStory(
        dataset._dataset_description,
        dataset._input_description,
        dataset._output_description,
        str(location),
        prediction,
        attributions=attributions,
    )

    return story.get_story_stream()


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

    tab1, tab2 = st.tabs(['Features', 'XAI methods'])

    def handle_tab1():
        row4_col1, row4_col2 = st.columns([0.5, 0.5])
        row5 = st.columns([0.3, 0.3, 0.3])
        row6 = st.columns([0.3, 0.3, 0.3])

        def write_explanation():
            index = gdf[
                (gdf['STATE_NAME'] == state) & (gdf['NAME'] == county)
            ].index[0]
            attributions = get_attributions(
                index,
                [attribution_method],
                [
                    get_channel_attributions,
                    get_timeseries_attributions,
                    get_multivariate_attributions,
                ],
            )
            channel_attributions_plot = get_channel_chart(
                attributions[0]['attributions'][0]
            )
            timeseries_attribution_plot = get_timeseries_chart(
                attributions[1]['attributions'][0]
            )
            multivariate_attribution_plot = get_multivariate_chart(
                [attributions[2]]
            )
            charts = (
                channel_attributions_plot,
                timeseries_attribution_plot,
                multivariate_attribution_plot,
            )
            for row, chart in zip(row5, charts):
                with row:
                    st.altair_chart(chart)
            futures = []
            with ThreadPoolExecutor() as executor:
                for attribution, column_stream in zip(attributions, row6):
                    ctx = get_script_run_ctx()
                    futures.append(
                        executor.submit(
                            stream_story_in_column,
                            ctx,
                            column_stream,
                            [attribution],
                            index,
                        )
                    )

        with row4_col2:
            attribution_method = st.selectbox(
                'Attribution method', ATTRIBUTION_METHODS
            )
        with row4_col1:
            st.button(
                'Generate explanations', key='tab1', on_click=write_explanation
            )

    def handle_tab2():
        row4_col1, row4_col2 = st.columns([0.5, 0.5])
        row5 = st.columns([0.9])[0]
        row6 = st.columns([0.3, 0.3, 0.3])
        row7 = st.columns(1)[0]

        # row6_col1 = st.columns(1)[0]
        def write_explanation():
            for method, col in zip(ATTRIBUTION_METHODS, row6):
                with col:
                    st.title(method)
            index = gdf[
                (gdf['STATE_NAME'] == state) & (gdf['NAME'] == county)
            ].index[0]
            attributions = get_attributions(
                index,
                ATTRIBUTION_METHODS,
                [get_multivariate_attributions],
            )
            chart = get_multivariate_chart(attributions)
            with row5:
                st.altair_chart(chart, use_container_width=False)
            futures = []
            with ThreadPoolExecutor() as executor:
                for i, (attribution, column) in enumerate(
                    zip(attributions, row6)
                ):
                    ctx = get_script_run_ctx()
                    futures.append(
                        executor.submit(
                            stream_story_in_column,
                            ctx,
                            column,
                            [attribution],
                            index,
                        )
                    )
            stories = [future.result() for future in futures]
            stream_comparison_in_column(row7, stories, index)

        with row4_col1:
            st.button(
                'Generate explanations', key='tab2', on_click=write_explanation
            )

    with tab1:
        handle_tab1()
    with tab2:
        handle_tab2()


def stream_comparison_in_column(column, stories, index) -> str:
    prompt, stream = get_comparison(stories, index)
    with column:
        with st.expander('See prompt'):
            st.chat_message('human').write(prompt)
        response = st.chat_message('assistant').write_stream(
            stream_handler(stream)
        )
        return response


def stream_story_in_column(ctx, column, attributions, index) -> str:
    prompt, stream = get_story(attributions, index)
    add_script_run_ctx(current_thread(), ctx)
    with column:
        with st.expander('See prompt'):
            st.chat_message('human').write(prompt)
        response = st.chat_message('assistant').write_stream(
            stream_handler(stream)
        )
        # accuracy, completeness = get_grades(response, attributions, index)
        # accuracy_prompt, stream = accuracy
        # accuracy = st.chat_message('assistant').write_stream(
        #    stream_handler(stream)
        # )

        # completeness_prompt, stream = completeness
        # with st.expander('See prompt'):
        #    st.chat_message('human').write(completeness_prompt)
        # completeness = st.chat_message('assistant').write_stream(
        #    stream_handler(stream)
        # )
    return response


def stream_handler(stream):
    for chunk in stream:
        yield chunk['message']['content']


app()
