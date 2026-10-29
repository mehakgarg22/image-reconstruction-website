import io
import os
import zipfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity


st.set_page_config(
    page_title="ImageRestore Studio",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%); }
      .block-container { max-width: 1250px; padding-top: 2rem; padding-bottom: 3rem; }
      .hero {
        padding: 1.6rem 1.8rem; border-radius: 20px; color: white;
        background: linear-gradient(120deg, #172554 0%, #1d4ed8 60%, #0e7490 100%);
        margin-bottom: 1.2rem;
      }
      .hero h1 { margin: 0 0 .45rem 0; font-size: 2.25rem; }
      .hero p { margin: 0; opacity: .92; font-size: 1.02rem; }
      .note {
        background: #fff7ed; border: 1px solid #fed7aa; color: #7c2d12;
        border-radius: 12px; padding: .85rem 1rem; margin: .6rem 0 1rem 0;
      }
      div[data-testid="stMetric"] {
        background: white; border: 1px solid #e2e8f0; padding: 12px 15px;
        border-radius: 14px;
      }
      .small-muted { color: #64748b; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🖼️ ImageRestore Studio</h1>
      <p>Upload a photograph, adjust restoration settings, and compare the original with the processed result.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Image helpers ----------
def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def auto_damage_mask(bgr: np.ndarray, sensitivity: int, detect_specks: bool) -> np.ndarray:
    """
    Conservative heuristic mask for likely bright/dark scratches and isolated specks.
    This cannot reliably identify every missing or damaged region in an arbitrary photo.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Local background estimate; high local contrast can indicate a scratch or speck.
    kernel_size = 15
    local = cv2.medianBlur(gray, kernel_size)
    difference = cv2.absdiff(gray, local)
    threshold = max(18, 60 - int(sensitivity * 0.42))
    mask = np.where(difference > threshold, 255, 0).astype(np.uint8)

    # Long, thin components are more likely to be scratches than image content.
    horizontal = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    )
    vertical = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    )
    scratch_mask = cv2.bitwise_or(horizontal, vertical)

    if detect_specks:
        # Include small isolated bright/dark specks using a blackhat/top-hat response.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
        specks = cv2.threshold(
            cv2.max(top_hat, black_hat), max(10, 36 - int(sensitivity * 0.25)), 255, cv2.THRESH_BINARY
        )[1]
        specks = cv2.morphologyEx(specks, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        scratch_mask = cv2.bitwise_or(scratch_mask, specks)

    # Slightly expand the mask so the inpainting algorithm covers scratch edges.
    scratch_mask = cv2.dilate(scratch_mask, np.ones((3, 3), np.uint8), iterations=1)
    return scratch_mask


def restore_image(
    original: np.ndarray,
    denoise_strength: int,
    contrast: float,
    sharpen: float,
    repair_scratches: bool,
    mask_sensitivity: int,
    inpaint_radius: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    working = original.copy()
    mask = None

    if repair_scratches:
        mask = auto_damage_mask(working, mask_sensitivity, detect_specks=True)
        # Avoid allowing the heuristic to alter a very large proportion of the photograph.
        masked_fraction = float(np.count_nonzero(mask)) / mask.size
        if masked_fraction > 0.18:
            # Conservative fallback: use only long thin scratch candidates.
            gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
            local = cv2.medianBlur(gray, 15)
            difference = cv2.absdiff(gray, local)
            raw = np.where(difference > max(22, 58 - mask_sensitivity // 2), 255, 0).astype(np.uint8)
            h = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
            v = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11)))
            mask = cv2.dilate(cv2.bitwise_or(h, v), np.ones((3, 3), np.uint8), iterations=1)
        if np.count_nonzero(mask) > 0:
            working = cv2.inpaint(working, mask, float(inpaint_radius), cv2.INPAINT_TELEA)

    # Denoising is optional through a strength control.
    if denoise_strength > 0:
        working = cv2.fastNlMeansDenoisingColored(
            working, None, float(denoise_strength), float(denoise_strength), 7, 21
        )

    # Contrast adjustment in LAB color space.
    if abs(contrast - 1.0) > 0.001:
        lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=max(1.0, float(contrast) * 2.0), tileGridSize=(8, 8))
        adjusted_l = clahe.apply(l_channel)
        # Blend according to selected contrast value; values below 1 are more subtle.
        blend = min(1.0, max(0.0, (contrast - 0.7) / 1.3))
        l_channel = cv2.addWeighted(l_channel, 1.0 - blend, adjusted_l, blend, 0)
        working = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

    # Unsharp masking for modest edge enhancement.
    if sharpen > 0:
        blurred = cv2.GaussianBlur(working, (0, 0), 1.2)
        working = cv2.addWeighted(working, 1.0 + float(sharpen), blurred, -float(sharpen), 0)

    return np.clip(working, 0, 255).astype(np.uint8), mask


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def compute_metrics(original: np.ndarray, restored: np.ndarray) -> dict:
    # These compare the output with the uploaded image, not with an unknown true original.
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    restored_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    mse = float(np.mean((original_rgb.astype(np.float32) - restored_rgb.astype(np.float32)) ** 2))
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(10 * np.log10((255.0 ** 2) / mse))
    ssim = float(structural_similarity(original_rgb, restored_rgb, channel_axis=2, data_range=255))
    return {"MSE": mse, "PSNR (dB)": psnr, "SSIM": ssim}


# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("⚙️ Restoration settings")
    st.caption("Processing starts automatically after upload or a settings change.")

    denoise_strength = st.slider("Denoising strength", 0, 15, 5, help="Higher values smooth more noise but may soften fine details.")
    contrast = st.slider("Contrast enhancement", 0.7, 2.0, 1.0, 0.1)
    sharpen = st.slider("Sharpening", 0.0, 1.5, 0.35, 0.05)
    st.divider()
    repair_scratches = st.toggle("Attempt automatic scratch / speck repair", value=True)
    mask_sensitivity = st.slider(
        "Damage detection sensitivity", 0, 100, 45,
        help="This is a heuristic detector. Increase carefully; image texture can be mistaken for damage.",
        disabled=not repair_scratches,
    )
    inpaint_radius = st.slider(
        "Repair radius", 1, 10, 3,
        help="Controls how much nearby image information is used to fill detected marks.",
        disabled=not repair_scratches,
    )
    st.divider()
    st.caption("Supported formats: JPG, JPEG, PNG, WEBP. Maximum upload size: 20 MB.")

# ---------- Upload and processing ----------
uploaded_file = st.file_uploader(
    "Upload your original photograph",
    type=["jpg", "jpeg", "png", "webp"],
    help="For best results, use a clear image with scratches, specks, noise, or mild degradation.",
)

if uploaded_file is None:
    st.info("Upload a photograph to automatically generate the original-versus-reconstructed comparison.")
    st.markdown(
        """
        **What this version can do**
        - Reduce visible noise and improve contrast and sharpness.
        - Attempt to repair some thin scratches and small specks automatically.
        - Preserve the uploaded image dimensions.

        **Important limitation:** completely missing objects, faces, or large image regions cannot be reliably inferred by these classical image-processing methods. That requires an AI inpainting model and usually a damage mask.
        """
    )
    st.stop()

try:
    pil_original = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
except Exception as exc:
    st.error(f"Could not read this image. Please upload a valid JPG, PNG, or WEBP file. Details: {exc}")
    st.stop()

width, height = pil_original.size
if width * height > 24_000_000:
    st.error("This image is larger than 24 megapixels. Please upload a smaller image to avoid running out of memory.")
    st.stop()

original_bgr = pil_to_bgr(pil_original)

with st.spinner("Reconstructing your image…"):
    restored_bgr, detected_mask = restore_image(
        original_bgr,
        denoise_strength=denoise_strength,
        contrast=contrast,
        sharpen=sharpen,
        repair_scratches=repair_scratches,
        mask_sensitivity=mask_sensitivity,
        inpaint_radius=inpaint_radius,
    )

restored_pil = bgr_to_pil(restored_bgr)
metrics = compute_metrics(original_bgr, restored_bgr)

st.success("Image processing complete.")
st.caption(f"Image dimensions preserved: {width} × {height} pixels")

original_col, restored_col = st.columns(2, gap="large")
with original_col:
    st.subheader("1. Original image")
    st.image(pil_original, use_container_width=True)
    st.download_button(
        "⬇️ Download original",
        data=image_to_bytes(pil_original),
        file_name="original_image.png",
        mime="image/png",
        use_container_width=True,
    )

with restored_col:
    st.subheader("2. Reconstructed image")
    st.image(restored_pil, use_container_width=True)
    st.download_button(
        "⬇️ Download reconstructed image",
        data=image_to_bytes(restored_pil),
        file_name="reconstructed_image.png",
        mime="image/png",
        use_container_width=True,
        type="primary",
    )

st.divider()
st.subheader("Restoration quality indicators")
st.caption("These metrics compare the processed image to the uploaded image. They do not prove that missing details were recovered, because the true undamaged image is unknown.")
metric_cols = st.columns(3)
metric_cols[0].metric("MSE", f"{metrics['MSE']:.2f}")
metric_cols[1].metric("PSNR", "∞" if np.isinf(metrics["PSNR (dB)"]) else f"{metrics['PSNR (dB)']:.2f} dB")
metric_cols[2].metric("SSIM", f"{metrics['SSIM']:.4f}")

if repair_scratches and detected_mask is not None:
    with st.expander("View automatic damage-detection mask"):
        st.caption("White areas are the pixels the heuristic detector attempted to repair. Check this mask because texture or fine detail may be incorrectly detected as damage.")
        st.image(Image.fromarray(detected_mask), caption="Detected candidate marks", use_container_width=True)

# Package outputs in one ZIP.
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
    zip_file.writestr("original_image.png", image_to_bytes(pil_original))
    zip_file.writestr("reconstructed_image.png", image_to_bytes(restored_pil))
    if detected_mask is not None:
        zip_file.writestr("detected_damage_mask.png", image_to_bytes(Image.fromarray(detected_mask)))
    report = [
        "ImageRestore Studio - Processing report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Image dimensions: {width} x {height}",
        f"Denoising strength: {denoise_strength}",
        f"Contrast enhancement: {contrast}",
        f"Sharpening: {sharpen}",
        f"Automatic scratch repair enabled: {repair_scratches}",
        f"Damage detection sensitivity: {mask_sensitivity}",
        f"Repair radius: {inpaint_radius}",
        f"MSE vs uploaded image: {metrics['MSE']:.6f}",
        f"PSNR vs uploaded image: {metrics['PSNR (dB)']}",
        f"SSIM vs uploaded image: {metrics['SSIM']:.6f}",
        "",
        "Note: Metrics compare the processed image with the uploaded image, not an unknown pristine reference.",
        "Classical image processing cannot reliably recover completely missing image content.",
    ]
    zip_file.writestr("processing_report.txt", "\n".join(report))

st.download_button(
    "📦 Download all results (ZIP)",
    data=zip_buffer.getvalue(),
    file_name="image_restore_results.zip",
    mime="application/zip",
    use_container_width=True,
)

with st.expander("How to get the best results"):
    st.markdown(
        """
        - Start with the default settings, then adjust one control at a time.
        - Increase denoising for visible grain, but lower it if fine textures disappear.
        - Use sharpening gently; too much can create halos.
        - Automatic scratch detection is approximate and may flag real texture as damage.
        - For large missing areas or missing facial/object details, use an AI inpainting model with a suitable damage mask. This app does not invent or guarantee recovery of information that is absent from the pixels.
        """
    )
