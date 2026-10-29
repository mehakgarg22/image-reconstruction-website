import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import richardson_lucy


# -----------------------------
# Page setup and visual styling
# -----------------------------
st.set_page_config(
    page_title="ImageRestore Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #64748b;
        --line: #e5eaf2;
        --accent: #635bff;
        --accent2: #8b5cf6;
        --surface: #ffffff;
        --soft: #f6f7fb;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(99,91,255,.08), transparent 35%),
            linear-gradient(180deg, #fbfcff 0%, #f7f8fc 100%);
        color: var(--ink);
    }
    .block-container {max-width: 1400px; padding-top: 1.5rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(99,91,255,.15);
        border-radius: 24px;
        background: linear-gradient(120deg, #ffffff 0%, #f0efff 58%, #f8faff 100%);
        box-shadow: 0 12px 40px rgba(31,41,55,.05);
        margin-bottom: 1.3rem;
    }
    .eyebrow {
        text-transform: uppercase; letter-spacing: .16em; font-size: .72rem;
        color: #635bff; font-weight: 800; margin-bottom: .5rem;
    }
    .hero h1 {font-size: clamp(2rem, 4vw, 3.1rem); line-height: 1.08; margin: 0; color: #151b31;}
    .hero p {font-size: 1.02rem; color: #5b6477; max-width: 780px; margin: .85rem 0 0;}
    .section-head {margin: 1.4rem 0 .4rem; font-size: 1.35rem; font-weight: 750; color: #172033;}
    .subtle {color: #64748b; font-size: .92rem;}
    .mini-card {
        background: white; border: 1px solid #e5eaf2; border-radius: 16px;
        padding: 1rem 1.1rem; min-height: 96px;
    }
    .mini-label {font-size: .78rem; color: #64748b; font-weight: 650;}
    .mini-value {font-size: 1.5rem; font-weight: 800; color: #172033; margin-top: .2rem;}
    .pill {
        display: inline-block; border-radius: 999px; padding: .3rem .65rem;
        background: #eeecff; color: #5146e5; font-size: .78rem; font-weight: 700;
    }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e5eaf2; border-radius: 14px;
        padding: .8rem 1rem;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #635bff, #8b5cf6);
        border: 0; border-radius: 12px; min-height: 3rem; font-weight: 750;
    }
    div.stDownloadButton > button {border-radius: 10px; font-weight: 650;}
    [data-testid="stFileUploader"] {
        background: white; border: 1px dashed #b7b8f8; border-radius: 16px; padding: .5rem;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Computer vision · Image restoration</div>
      <h1>ImageRestore Studio</h1>
      <p>Clean up noise, enhance details, experiment with blur restoration, and repair
      a selected damaged region — with side-by-side previews and downloadable results.</p>
    </div>
    """,
    unsafe_allow_html=True,
