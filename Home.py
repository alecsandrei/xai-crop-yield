from __future__ import annotations

import streamlit as st

st.set_page_config(layout='wide')

st.title("Using LLM's for XAI")

st.header('Instructions')

markdown = """
1. Go to the **Soybean crop yield XAI** tab on the left;
2. Choose a county;
3. Choose the XAI method;
4. Click on **Generate explanations**.
"""

st.markdown(markdown)

st.video(
    'https://raw.githubusercontent.com/alecsandrei/xai-crop-yield/master/docs/example.mp4',
    autoplay=True,
    loop=True,
    muted=True,
)
