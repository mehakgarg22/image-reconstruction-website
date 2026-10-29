
import io
import zipfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import richardson_lucy

# ============================================================
# PAGE CONFIGURATION
# ============================================================
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
    }
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(99,91,255,.08), transparent 35%),
            linear-gradient(180deg, #fbfcff 0%, #f7f8fc 100%);
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
        background: linear-gradient(120deg, #fff 0%, #f0efff 58%, #f8faff 100%);
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
    .subtle { color: #64748b; font-size: .92rem; }
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
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Computer vision · Image restoration</div>
      <h1>ImageRestore Studio</h1>
      <p>
        Remove simulated noise, restore blurred images, and repair a damaged
        region. Explore every intermediate step, compare results, and
        download your images.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
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
                "Please upload a smaller image. The app does not "
                "resize images silently."
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
            reference, result, data_range=255
        )
    )

    ssim = float(
        structural_similarity(
            reference,
            result,
            channel_axis=-1,
            data_range=255,
        )
    )

    return mse, psnr, ssim


def unsharp_mask(image, amount=0.5, radius=1.2):
    blurred = cv2.GaussianBlur(
        image, (0, 0), sigmaX=radius, sigmaY=radius
    )

    result = cv2.addWeighted(
        image.astype(np.float32),
        1.0 + amount,
        blurred.astype(np.float32),
        -amount,
        0,
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def denoise_image(image, strength=5):
    h = int(strength)

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h,
        max(3, int(round(h * 0.9))),
        7,
        21,
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
    kernel = make_blur_kernel(
        int(kernel_size), float(sigma)
    )

    channels = []

    for channel in cv2.split(
        image.astype(np.float32) / 255.0
    ):
        restored = richardson_lucy(
            channel,
            kernel,
            num_iter=int(iterations),
            clip=True,
        )

        channels.append(
            np.clip(restored * 255.0, 0, 255)
        )

    return cv2.merge(channels).astype(np.uint8)


def compare_metric_rows(reference, restored_images):
    rows = []

    for name, result in restored_images.items():
        mse, psnr, ssim = evaluate(reference, result)

        rows.append({
            "Experiment": name,
            "MSE ↓": round(mse, 3),
            "PSNR (dB) ↑": round(psnr, 3),
            "SSIM ↑": round(ssim, 4),
        })

    return pd.DataFrame(rows)


def build_results_zip(original, stages, metrics):
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(
            "original.png", to_png_bytes(original)
        )

        for method, steps in stages.items():
            folder = method.lower().replace(" ", "_")

            for index, (step_name, image) in enumerate(
                steps, start=1
            ):
                filename = (
                    step_name.lower()
                    .replace(" ", "_")
                    .replace(":", "")
                )

                archive.writestr(
                    f"{folder}/{index}_{filename}.png",
                    to_png_bytes(image),
                )

        archive.writestr(
            "quality_metrics.csv",
            metrics.to_csv(index=False),
        )

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SIDEBAR SETTINGS
# ============================================================
with st.sidebar:
    st.markdown("## ✦ Workspace")
    st.caption(
        "Upload an image to automatically run the reconstruction steps."
    )

    uploaded_file = st.file_uploader(
        "Upload your photograph",
        type=["jpg", "jpeg", "png", "webp"],
        help=(
            f"Maximum file size: {MAX_FILE_MB} MB. "
            "Images are not silently resized."
        ),
    )

    st.markdown("---")
    st.markdown("### Restoration settings")

    denoise_strength = st.slider(
        "Denoising strength", 3, 12, 5
    )

    sharpen_amount = st.slider(
        "Detail enhancement", 0.0, 1.5, 0.35, 0.05
    )

    sharpen_radius = st.slider(
        "Sharpen radius", 0.5, 3.0, 1.2, 0.1
    )

    noise_strength = st.slider(
        "Synthetic noise level", 5, 60, 25
    )

    blur_size = st.select_slider(
        "Synthetic blur kernel",
        options=[3, 5, 7, 9, 11, 13],
        value=9,
    )

    blur_sigma = st.slider(
        "Deblur blur estimate", 0.8, 3.0, 2.0, 0.1
    )

    deblur_iterations = st.slider(
        "Deblur iterations", 5, 25, 10, 5
    )

    missing_percent = st.slider(
        "Simulated missing region (%)", 5, 30, 15
    )

    st.caption(
        "Changing a setting automatically reruns the experiments."
    )


# ============================================================
# EMPTY STATE
# ============================================================
if uploaded_file is None:
    st.markdown("### Start with a photograph")

    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        st.markdown(
            """
            **Your workflow**

            1. Upload a JPG, PNG, or WebP image.
            2. View the original photograph.
            3. Inspect every intermediate processing step.
            4. Compare reconstructed images and quality metrics.
            5. Download individual images or a complete ZIP report.
            """
        )

        st.info(
            "For clean photos, use low denoising and sharpening settings. "
            "Restoration is not guaranteed to improve every image."
        )

    with right:
        st.markdown(
            """
            <div class="mini-card">
              <div class="mini-label">RESTORATION EXPERIMENTS</div>
              <div class="mini-value">03</div>
              <p class="subtle">
                Noise removal · Deblurring · Missing-region repair
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================
try:
    original = load_rgb(uploaded_file)

except (ValueError, UnidentifiedImageError, OSError) as exc:
    if isinstance(exc, ValueError):
        st.error(str(exc))
    else:
        st.error(
            "Could not read this image. "
            "Try a valid JPG, PNG, or WebP file."
        )

    st.stop()


height, width = original.shape[:2]

st.markdown(
    f'<span class="pill">Image loaded</span> &nbsp; '
    f'{width:,} × {height:,} pixels',
    unsafe_allow_html=True,
)

original_tab, pipeline_tab, metrics_tab, guide_tab = st.tabs([
    "① Original image",
    "② Step-by-step pipeline",
    "③ Results & metrics",
    "④ How it works",
])


# ============================================================
# ORIGINAL IMAGE TAB
# ============================================================
with original_tab:
    st.markdown(
        '<div class="section-head">Your original photograph</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "The image dimensions are preserved. Browser display scaling "
        "does not change the downloaded image."
    )

    st.image(original, use_container_width=True)

    st.download_button(
        "Download original PNG",
        data=to_png_bytes(original),
        file_name="original_image.png",
        mime="image/png",
    )


# ============================================================
# AUTOMATIC IMAGE PROCESSING
# ============================================================
try:
    with st.spinner(
        "Processing your image and preparing intermediate results..."
    ):
        # A. NOISE REMOVAL
        # Original -> noisy -> denoised -> enhanced
        rng = np.random.default_rng(42)

        noise = rng.normal(
            0, noise_strength, original.shape
        )

        noisy = np.clip(
            original.astype(np.float32) + noise,
            0,
            255,
        ).astype(np.uint8)

        denoised_before_sharpening = denoise_image(
            noisy, denoise_strength
        )

        if sharpen_amount > 0:
            denoised = unsharp_mask(
                denoised_before_sharpening,
                sharpen_amount,
                sharpen_radius,
            )
        else:
            denoised = denoised_before_sharpening.copy()

        # B. BLUR RESTORATION
        # Original -> blurred -> deblurred -> enhanced
        blurred = cv2.GaussianBlur(
            original, (int(blur_size), int(blur_size)), 0
        )

        deblurred_before_sharpening = deblur_image(
            blurred,
            iterations=deblur_iterations,
            kernel_size=int(blur_size),
            sigma=blur_sigma,
        )

        if sharpen_amount > 0:
            deblurred = unsharp_mask(
                deblurred_before_sharpening,
                sharpen_amount * 0.35,
                sharpen_radius,
            )
        else:
            deblurred = deblurred_before_sharpening.copy()

        # C. MISSING-REGION REPAIR
        # Original -> damaged -> repaired
        damaged = original.copy()

        mask = np.zeros(
            (height, width), dtype=np.uint8
        )

        patch_w = max(
            1, int(width * missing_percent / 100)
        )

        patch_h = max(
            1, int(height * missing_percent / 100)
        )

        x1 = (width - patch_w) // 2
        y1 = (height - patch_h) // 2
        x2 = x1 + patch_w
        y2 = y1 + patch_h

        damaged[y1:y2, x1:x2] = 0
        mask[y1:y2, x1:x2] = 255

        inpainted = cv2.inpaint(
            damaged, mask, 3, cv2.INPAINT_TELEA
        )

        # Intermediate images shown on the website
        stages = {
            "Noise Removal": [
                ("Step 1: Original Image", original),
                ("Step 2: Synthetic Noise Added", noisy),
                (
                    "Step 3: After Denoising",
                    denoised_before_sharpening,
                ),
                ("Step 4: Final Enhanced Image", denoised),
            ],
            "Blur Restoration": [
                ("Step 1: Original Image", original),
                ("Step 2: Synthetic Blur Applied", blurred),
                (
                    "Step 3: After Deblurring",
                    deblurred_before_sharpening,
                ),
                ("Step 4: Final Restored Image", deblurred),
            ],
            "Missing Region Repair": [
                ("Step 1: Original Image", original),
                ("Step 2: Region Removed", damaged),
                ("Step 3: Region Repaired", inpainted),
            ],
        }

        degraded = {
            "Noise Removal": noisy,
            "Blur Restoration": blurred,
            "Missing Region Repair": damaged,
        }

        restored = {
            "Noise Removal": denoised,
            "Blur Restoration": deblurred,
            "Missing Region Repair": inpainted,
        }

        metrics = compare_metric_rows(
            original, restored
        )

except Exception as exc:
    st.error(f"Image processing failed: {exc}")
    st.info(
        "Try a smaller image or reduce the deblurring iterations "
        "in the sidebar."
    )
    st.stop()


# ============================================================
# STEP-BY-STEP PIPELINE
# ============================================================
with pipeline_tab:
    st.markdown(
        '<div class="section-head">'
        'Step-by-step reconstruction'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Choose a method below. Each image shows what happens "
        "at that stage of the process."
    )

    selected_method = st.selectbox(
        "Choose reconstruction method",
        list(stages.keys()),
        key="pipeline_method",
    )

    selected_steps = stages[selected_method]

    for start in range(0, len(selected_steps), 2):
        cols = st.columns(2, gap="large")

        for offset, col in enumerate(cols):
            idx = start + offset

            if idx >= len(selected_steps):
                continue

            step_title, step_image = selected_steps[idx]

            with col:
                st.markdown(f"**{step_title}**")

                st.image(
                    step_image,
                    use_container_width=True,
                )

                st.download_button(
                    f"Download image — step {idx + 1}",
                    data=to_png_bytes(step_image),
                    file_name=(
                        f"{selected_method.lower().replace(' ', '_')}"
                        f"_step_{idx + 1}.png"
                    ),
                    mime="image/png",
                    key=f"pipeline_download_{selected_method}_{idx}",
                    use_container_width=True,
                )


# ============================================================
# RESULTS AND METRICS
# ============================================================
with metrics_tab:
    st.markdown(
        '<div class="section-head">'
        'Original vs. reconstructed image'
        '</div>',
        unsafe_allow_html=True,
    )

    method_for_comparison = st.selectbox(
        "Choose result to compare",
        list(restored.keys()),
        key="result_method",
    )

    compare_left, compare_right = st.columns(
        2, gap="large"
    )

    with compare_left:
        st.markdown("**Degraded input**")
        st.image(
            degraded[method_for_comparison],
            use_container_width=True,
        )

    with compare_right:
        st.markdown("**Final reconstructed output**")
        st.image(
            restored[method_for_comparison],
            use_container_width=True,
        )

    st.markdown(
        '<div class="section-head">Quality evaluation</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Metrics compare each reconstructed image with the uploaded "
        "original. They may not perfectly reflect perceived quality."
    )

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True,
    )

    row = metrics[
        metrics["Experiment"] == method_for_comparison
    ].iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MSE",
        f'{row["MSE ↓"]:.3f}',
        help="Lower is generally better.",
    )

    c2.metric(
        "PSNR",
        f'{row["PSNR (dB) ↑"]:.2f} dB',
        help="Higher is generally better.",
    )

    c3.metric(
        "SSIM",
        f'{row["SSIM ↑"]:.4f}',
        help="Closer to 1 generally indicates greater structural similarity.",
    )

    st.markdown(
        '<div class="section-head">Download results</div>',
        unsafe_allow_html=True,
    )

    download_cols = st.columns(3)

    for idx, (name, image) in enumerate(restored.items()):
        with download_cols[idx]:
            st.download_button(
                f"Download {name}",
                data=to_png_bytes(image),
                file_name=(
                    f"{name.lower().replace(' ', '_')}"
                    "_reconstructed.png"
                ),
                mime="image/png",
                key=f"final_download_{idx}",
                use_container_width=True,
            )

    zip_data = build_results_zip(
        original, stages, metrics
    )

    st.download_button(
        "Download all steps and metrics (ZIP)",
        data=zip_data,
        file_name="imagerestore_complete_results.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

    st.success(
        "Processing complete. All intermediate and final images "
        "are available in the pipeline tab."
    )


# ============================================================
# HOW IT WORKS
# ============================================================
with guide_tab:
    st.markdown("### Restoration methods")

    st.markdown(
        """
        **1. Noise removal**

        Random noise is added for demonstration, then OpenCV
        Non-Local Means denoising reduces it. A final detail
        enhancement step is optional.

        **2. Blur restoration**

        Gaussian blur is applied to the original image.
        Richardson–Lucy deconvolution attempts to reverse that
        blur using an estimated kernel. Results depend on how
        closely the estimate matches the actual blur.

        **3. Missing-region repair**

        A rectangular patch is removed from the center.
        OpenCV Telea inpainting fills the region using nearby
        pixels. It cannot know the exact original contents of
        the missing area.

        ### Metric interpretation

        - **MSE:** Mean squared pixel error; lower is generally better.
        - **PSNR:** Signal-to-noise ratio relative to the reference;
          higher is generally better.
        - **SSIM:** Structural similarity; values closer to 1 generally
          indicate greater similarity.

        These metrics use the uploaded image as a known reference
        because the app creates synthetic damage from it.
        """
    )

    st.warning(
        "This project demonstrates traditional image-processing "
        "techniques, not an AI model trained to recover every lost "
        "detail. Some results may look worse than the original."
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.caption(
    "ImageRestore Studio · Educational computer-vision project · "
    "Uploaded image dimensions preserved"
)
