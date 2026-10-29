
import io
import zipfile

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
            radial-gradient(
                ellipse at 8% 0%,
                rgba(99,91,255,.08),
                transparent 35%
            ),
            linear-gradient(
                180deg,
                #fbfcff 0%,
                #f7f8fc 100%
            );
        color: var(--ink);
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(99,91,255,.15);
        border-radius: 24px;
        background: linear-gradient(
            120deg,
            #ffffff 0%,
            #f0efff 58%,
            #f8faff 100%
        );
        box-shadow: 0 12px 40px rgba(31,41,55,.05);
        margin-bottom: 1.3rem;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .72rem;
        color: #635bff;
        font-weight: 800;
        margin-bottom: .5rem;
    }

    .hero h1 {
        font-size: clamp(2rem, 4vw, 3.1rem);
        line-height: 1.08;
        margin: 0;
        color: #151b31;
    }

    .hero p {
        font-size: 1.02rem;
        color: #5b6477;
        max-width: 780px;
        margin: .85rem 0 0;
    }

    .section-head {
        margin: 1.4rem 0 .4rem;
        font-size: 1.35rem;
        font-weight: 750;
        color: #172033;
    }

    .subtle {
        color: #64748b;
        font-size: .92rem;
    }

    .mini-card {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 96px;
    }

    .mini-label {
        font-size: .78rem;
        color: #64748b;
        font-weight: 650;
    }

    .mini-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #172033;
        margin-top: .2rem;
    }

    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: .3rem .65rem;
        background: #eeecff;
        color: #5146e5;
        font-size: .78rem;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 14px;
        padding: .8rem 1rem;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #635bff, #8b5cf6);
        border: 0;
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 750;
    }

    div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #b7b8f8;
        border-radius: 16px;
        padding: .5rem;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">
        Computer vision · Image restoration
      </div>
      <h1>ImageRestore Studio</h1>
      <p>
        Clean up noise, enhance details, experiment with blur restoration,
        and repair a selected damaged region — with side-by-side previews
        and downloadable results.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helper functions
# -----------------------------
MAX_FILE_MB = 20
MAX_PIXELS = 24_000_000


def load_rgb(uploaded_file):
    raw = uploaded_file.getvalue()

    if len(raw) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(
            f"Please upload an image smaller than {MAX_FILE_MB} MB."
        )

    with Image.open(io.BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")

        if image.width * image.height > MAX_PIXELS:
            raise ValueError(
                "This image has more than 24 megapixels. "
                "Please upload a smaller image to keep processing responsive. "
                "The app does not resize images silently."
            )

        return np.asarray(image, dtype=np.uint8)


def to_png_bytes(array):
    buffer = io.BytesIO()

    Image.fromarray(
        np.asarray(array, dtype=np.uint8)
    ).save(buffer, format="PNG")

    return buffer.getvalue()


def evaluate(reference, result):
    reference = reference.astype(np.uint8)
    result = result.astype(np.uint8)

    mse = float(
        np.mean(
            (
                reference.astype(np.float32)
                - result.astype(np.float32)
            ) ** 2
        )
    )

    psnr = float(
        peak_signal_noise_ratio(
            reference,
            result,
            data_range=255
        )
    )

    ssim = float(
        structural_similarity(
            reference,
            result,
            channel_axis=-1,
            data_range=255
        )
    )

    return mse, psnr, ssim


def unsharp_mask(image, amount=0.7, radius=1.2):
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=radius,
        sigmaY=radius
    )

    result = cv2.addWeighted(
        image.astype(np.float32),
        1.0 + amount,
        blurred.astype(np.float32),
        -amount,
        0
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def denoise_image(image, strength=5):
    h = int(strength)
    h_color = max(3, int(round(h * 0.9)))

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h,
        h_color,
        7,
        21
    )


def make_blur_kernel(size=9, sigma=2.0):
    coords = (
        np.arange(size, dtype=np.float32)
        - (size - 1) / 2
    )

    xx, yy = np.meshgrid(coords, coords)

    kernel = np.exp(
        -(xx * xx + yy * yy) / (2 * sigma * sigma)
    )

    kernel /= kernel.sum()

    return kernel


def deblur_image(image, iterations=10, kernel_size=9, sigma=2.0):
    kernel = make_blur_kernel(kernel_size, sigma)
    channels = []

    for channel in cv2.split(
        image.astype(np.float32) / 255.0
    ):
        restored = richardson_lucy(
            channel,
            kernel,
            num_iter=int(iterations),
            clip=True
        )

        channels.append(
            np.clip(restored * 255.0, 0, 255)
        )

    return cv2.merge(channels).astype(np.uint8)


def compare_metric_rows(reference, pairs):
    rows = []

    for name, result in pairs.items():
        mse, psnr, ssim = evaluate(reference, result)

        rows.append({
            "Experiment": name,
            "MSE ↓": round(mse, 3),
            "PSNR (dB) ↑": round(psnr, 3),
            "SSIM ↑": round(ssim, 4),
        })

    return pd.DataFrame(rows)


def build_results_zip(original, degraded_results, restored_results, metrics):
    data = io.BytesIO()

    with zipfile.ZipFile(
        data,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        archive.writestr(
            "original.png",
            to_png_bytes(original)
        )

        for name, image in degraded_results.items():
            safe_name = name.lower().replace(" ", "_")

            archive.writestr(
                f"{safe_name}_degraded.png",
                to_png_bytes(image)
            )

        for name, image in restored_results.items():
            safe_name = name.lower().replace(" ", "_")

            archive.writestr(
                f"{safe_name}_reconstructed.png",
                to_png_bytes(image)
            )

        archive.writestr(
            "quality_metrics.csv",
            metrics.to_csv(index=False)
        )

    data.seek(0)

    return data.getvalue()


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.markdown("## ✦ Workspace")
    st.caption(
        "Upload once, then choose the restoration experiment."
    )

    uploaded_file = st.file_uploader(
        "Upload your photograph",
        type=["jpg", "jpeg", "png", "webp"],
        help=(
            f"Maximum file size: {MAX_FILE_MB} MB. "
            "Images are not silently resized."
        )
    )

    st.markdown("---")
    st.markdown("### Restoration settings")

    denoise_strength = st.slider(
        "Denoising strength",
        min_value=3,
        max_value=12,
        value=5,
        help=(
            "Start low to retain texture. Increase only "
            "if the input has strong noise."
        )
    )

    sharpen_amount = st.slider(
        "Detail enhancement",
        min_value=0.0,
        max_value=1.5,
        value=0.35,
        step=0.05,
        help=(
            "0 disables sharpening. Too much can create "
            "halos and exaggerate noise."
        )
    )

    sharpen_radius = st.slider(
        "Sharpen radius",
        min_value=0.5,
        max_value=3.0,
        value=1.2,
        step=0.1
    )

    noise_strength = st.slider(
        "Synthetic noise level",
        min_value=5,
        max_value=60,
        value=25
    )

    blur_size = st.select_slider(
        "Synthetic blur kernel",
        options=[3, 5, 7, 9, 11, 13],
        value=9
    )

    blur_sigma = st.slider(
        "Deblur blur estimate",
        min_value=0.8,
        max_value=3.0,
        value=2.0,
        step=0.1,
        help=(
            "This is an estimate of the blur kernel; "
            "actual blur may differ."
        )
    )

    deblur_iterations = st.slider(
        "Deblur iterations",
        min_value=5,
        max_value=25,
        value=10,
        step=5,
        help=(
            "More iterations can increase sharpness "
            "but also amplify artifacts."
        )
    )

    missing_percent = st.slider(
        "Simulated missing region (%)",
        min_value=5,
        max_value=30,
        value=15
    )

    run = st.button(
        "Run all experiments",
        type="primary",
        use_container_width=True
    )

    st.caption(
        "Privacy note: avoid uploading sensitive images to a public demo."
    )


# -----------------------------
# Empty state
# -----------------------------
if uploaded_file is None:
    st.markdown("### Start with a photograph")

    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        st.markdown(
            """
            **Your workflow**

            1. Upload a JPG, PNG, or WebP photo.
            2. Adjust denoising and sharpening strength.
            3. Run the three restoration experiments.
            4. Compare visual results and reference-based metrics.
            5. Download the results individually or as a ZIP.
            """
        )

        st.info(
            "For the clearest result, use a high-quality original photo. "
            "If your image is already sharp and clean, leave denoising "
            "low and use detail enhancement sparingly."
        )

    with right:
        st.markdown(
            """
            <div class="mini-card">
              <div class="mini-label">EXPERIMENTS INCLUDED</div>
              <div class="mini-value">03</div>
              <p class="subtle">
                Noise removal · Deblurring · Inpainting
              </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.stop()


# -----------------------------
# Load uploaded image
# -----------------------------
try:
    original = load_rgb(uploaded_file)

except (
    ValueError,
    UnidentifiedImageError,
    OSError
) as exc:
    if isinstance(exc, ValueError):
        st.error(str(exc))
    else:
        st.error(
            "This image could not be read. "
            "Try a valid JPG, PNG, or WebP file."
        )

    st.stop()


height, width = original.shape[:2]

st.markdown(
    f'<span class="pill">Image loaded</span> &nbsp; '
    f'{width:,} × {height:,} pixels '
    f'&nbsp; · &nbsp; RGB photograph',
    unsafe_allow_html=True
)


# -----------------------------
# Main tabs
# -----------------------------
preview_tab, results_tab, guide_tab = st.tabs(
    [
        "① Original image",
        "② Results & metrics",
        "③ How it works"
    ]
)


# -----------------------------
# Original image tab
# -----------------------------
with preview_tab:
    st.markdown(
        '<div class="section-head">Your original photograph</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "The app preserves the uploaded pixel dimensions. "
        "Display scaling in the browser does not change the downloaded file."
    )

    st.image(
        original,
        use_container_width=True
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.download_button(
            "Download original PNG",
            data=to_png_bytes(original),
            file_name="original_image.png",
            mime="image/png"
        )

    with col_b:
        st.metric(
            "Pixel dimensions",
            f"{width:,} × {height:,}"
        )


# -----------------------------
# Results and metrics tab
# -----------------------------
with results_tab:
    if not run:
        st.markdown("### Ready to restore")

        st.write(
            "Adjust the controls in the sidebar, then select "
            "**Run all experiments**."
        )

        st.info(
            "The app creates controlled degraded versions of your photo "
            "so it can compare each reconstruction with the known original."
        )

    else:
        try:
            with st.spinner("Running restoration experiments..."):

                # A. Add synthetic noise and remove it.
                rng = np.random.default_rng(42)

                noise = rng.normal(
                    0,
                    noise_strength,
                    original.shape
                )

                noisy = np.clip(
                    original.astype(np.float32) + noise,
                    0,
                    255
                ).astype(np.uint8)

                denoised = denoise_image(
                    noisy,
                    denoise_strength
                )

                if sharpen_amount > 0:
                    denoised = unsharp_mask(
                        denoised,
                        amount=sharpen_amount,
                        radius=sharpen_radius
                    )

                # B. Apply synthetic blur and attempt deblurring.
                blurred = cv2.GaussianBlur(
                    original,
                    (int(blur_size), int(blur_size)),
                    0
                )

                deblurred = deblur_image(
                    blurred,
                    iterations=deblur_iterations,
                    kernel_size=int(blur_size),
                    sigma=blur_sigma
                )

                if sharpen_amount > 0:
                    deblurred = unsharp_mask(
                        deblurred,
                        amount=sharpen_amount * 0.35,
                        radius=sharpen_radius
                    )

                # C. Simulate a missing rectangular region.
                damaged = original.copy()

                mask = np.zeros(
                    (height, width),
                    dtype=np.uint8
                )

                patch_w = max(
                    1,
                    int(width * missing_percent / 100)
                )

                patch_h = max(
                    1,
                    int(height * missing_percent / 100)
                )

                x1 = (width - patch_w) // 2
                y1 = (height - patch_h) // 2
                x2 = x1 + patch_w
                y2 = y1 + patch_h

                damaged[y1:y2, x1:x2] = 0
                mask[y1:y2, x1:x2] = 255

                inpainted = cv2.inpaint(
                    damaged,
                    mask,
                    3,
                    cv2.INPAINT_TELEA
                )

                degraded = {
                    "Noise Removal": noisy,
                    "Blur Restoration": blurred,
                    "Missing Region": damaged,
                }

                restored = {
                    "Noise Removal": denoised,
                    "Blur Restoration": deblurred,
                    "Missing Region": inpainted,
                }

                metrics = compare_metric_rows(
                    original,
                    restored
                )

            st.markdown(
                '<div class="section-head">'
                'Original vs. reconstructed'
                '</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "Compare each reconstruction with its degraded input "
                "and original reference. Use the sidebar settings to tune "
                "the output."
            )

            method_choice = st.selectbox(
                "Choose experiment to inspect",
                list(restored.keys()),
                key="method_choice"
            )

            dcol, rcol = st.columns(2, gap="large")

            with dcol:
                st.markdown("**Degraded input**")

                st.image(
                    degraded[method_choice],
                    use_container_width=True
                )

            with rcol:
                st.markdown("**Reconstructed output**")

                st.image(
                    restored[method_choice],
                    use_container_width=True
                )

            st.markdown(
                '<div class="section-head">Quality evaluation</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "Metrics compare each reconstructed image with the "
                "uploaded original. They are reference-based and may not "
                "perfectly reflect perceived quality."
            )

            st.dataframe(
                metrics,
                use_container_width=True,
                hide_index=True
            )

            selected_row = metrics[
                metrics["Experiment"] == method_choice
            ].iloc[0]

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "MSE",
                f'{selected_row["MSE ↓"]:.3f}',
                help="Lower is generally better."
            )

            m2.metric(
                "PSNR",
                f'{selected_row["PSNR (dB) ↑"]:.2f} dB',
                help="Higher is generally better."
            )

            m3.metric(
                "SSIM",
                f'{selected_row["SSIM ↑"]:.4f}',
                help=(
                    "Closer to 1 generally indicates greater "
                    "structural similarity."
                )
            )

            st.markdown(
                '<div class="section-head">Download your results</div>',
                unsafe_allow_html=True
            )

            download_cols = st.columns(3)
            method_names = list(restored.keys())

            for index, name in enumerate(method_names):
                with download_cols[index]:
                    st.download_button(
                        f"Download {name}",
                        data=to_png_bytes(restored[name]),
                        file_name=(
                            name.lower().replace(" ", "_")
                            + "_reconstructed.png"
                        ),
                        mime="image/png",
                        key=f"individual_{index}",
                        use_container_width=True
                    )

            zip_data = build_results_zip(
                original,
                degraded,
                restored,
                metrics
            )

            st.download_button(
                "Download complete report (ZIP)",
                data=zip_data,
                file_name="imagerestore_results.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )

            st.success(
                "Experiments completed. If a result looks over-smoothed, "
                "lower denoising strength. If edges look harsh, lower "
                "detail enhancement."
            )

        except Exception as exc:
            st.error(f"Processing failed: {exc}")

            st.caption(
                "Try a smaller image or lower the deblurring iteration count."
            )


# -----------------------------
# How it works tab
# -----------------------------
with guide_tab:
    st.markdown("### Restoration methods")

    st.markdown(
        """
        - **Noise removal:** Non-local means denoising reduces simulated
          random noise. Strong denoising may soften fine textures.

        - **Blur restoration:** Richardson–Lucy deconvolution estimates
          a sharper image from an estimated blur kernel. It works best
          for mild blur that resembles the assumed kernel and can produce
          ringing or noise when the estimate is incorrect.

        - **Missing-region repair:** OpenCV Telea inpainting estimates
          pixels in a simulated rectangular damaged region from nearby
          pixels. It cannot recover unknown original details exactly.
        """
    )

    st.markdown("### Metric interpretation")

    st.markdown(
        """
        - **MSE:** Mean squared pixel error; lower is better.
        - **PSNR:** Signal-to-noise ratio relative to the reference;
          higher is generally better.
        - **SSIM:** Structural similarity; values closer to 1 generally
          indicate greater similarity.

        Because the app creates the degraded input from the uploaded
        original, it has a reference image for evaluation. For a real
        damaged photo with no original reference, these metrics cannot
        be calculated in the same way.
        """
    )

    st.warning(
        "The app does not guarantee that every restoration will improve "
        "an already-clean photo. Use low denoising strength for clean "
        "images and compare outputs before downloading."
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "ImageRestore Studio · Educational image-processing demo · "
    "Original image dimensions are preserved"
)
