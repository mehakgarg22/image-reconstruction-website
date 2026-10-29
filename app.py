
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import zipfile

from io import BytesIO
from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

st.set_page_config(
    page_title="ImageRestore | Image Reconstruction",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ ImageRestore")
st.subheader("Image Reconstruction & Restoration")
st.write(
    "Upload a normal photograph to explore noise removal, "
    "blur restoration, and missing-region reconstruction."
)

st.info(
    "Your image is processed in the current app session. "
    "The algorithms below use traditional image processing, "
    "not a trained AI model."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Reconstruction functions
# -----------------------------

def evaluate_image(reference, result):
    reference = reference.astype(np.uint8)
    result = result.astype(np.uint8)

    mse = float(np.mean(
        (reference.astype(np.float32) -
         result.astype(np.float32)) ** 2
    ))

    psnr = float(peak_signal_noise_ratio(
        reference, result, data_range=255
    ))

    ssim = float(structural_similarity(
        reference,
        result,
        channel_axis=-1,
        data_range=255
    ))

    return mse, psnr, ssim


def create_experiments(original, noise_strength, blur_size,
                       missing_percent):
    rng = np.random.default_rng(42)
    experiments = {}

    # 1. Gaussian noise and denoising
    noise = rng.normal(
        0, noise_strength, original.shape
    )

    noisy = np.clip(
        original.astype(np.float32) + noise,
        0, 255
    ).astype(np.uint8)

    denoised = cv2.fastNlMeansDenoisingColored(
        noisy, None, 10, 10, 7, 21
    )

    experiments["Noise Removal"] = (
        noisy, denoised
    )

    # 2. Blur and sharpening
    blurred = cv2.GaussianBlur(
        original, (blur_size, blur_size), 0
    )

    smooth = cv2.GaussianBlur(
        blurred, (0, 0), 3
    )

    sharpened = cv2.addWeighted(
        blurred, 1.8, smooth, -0.8, 0
    )

    experiments["Blur Restoration"] = (
        blurred, sharpened
    )

    # 3. Missing region and inpainting
    damaged = original.copy()

    height, width = original.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    patch_w = max(1, int(width * missing_percent / 100))
    patch_h = max(1, int(height * missing_percent / 100))

    x1 = (width - patch_w) // 2
    y1 = (height - patch_h) // 2
    x2 = x1 + patch_w
    y2 = y1 + patch_h

    damaged[y1:y2, x1:x2] = 0
    mask[y1:y2, x1:x2] = 255

    inpainted = cv2.inpaint(
        damaged, mask, 5, cv2.INPAINT_TELEA
    )

    experiments["Missing Region"] = (
        damaged, inpainted
    )

    return experiments


def image_png_bytes(image):
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


# -----------------------------
# Main website
# -----------------------------

if uploaded_file is None:
    st.markdown("### How it works")
    st.markdown(
        "1. Upload a photograph.\n"
        "2. Generate degraded versions.\n"
        "3. Reconstruct the images.\n"
        "4. Compare quality metrics.\n"
        "5. Download the results."
    )
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((800, 800))
    original = np.array(image)
except Exception:
    st.error("Unable to read this image. Please upload a valid JPG or PNG.")
    st.stop()

st.success(
    f"Image loaded successfully: "
    f"{original.shape[1]} × {original.shape[0]} pixels"
)

with st.expander("Adjust reconstruction settings", expanded=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        noise_strength = st.slider(
            "Gaussian noise strength",
            min_value=5,
            max_value=60,
            value=25
        )

    with c2:
        blur_size = st.slider(
            "Blur kernel size",
            min_value=3,
            max_value=15,
            value=9,
            step=2
        )

    with c3:
        missing_percent = st.slider(
            "Missing region size (%)",
            min_value=5,
            max_value=30,
            value=15
        )

if st.button("Reconstruct My Image", type="primary"):
    with st.spinner("Processing your image..."):
        experiments = create_experiments(
            original,
            noise_strength,
            blur_size,
            missing_percent
        )

    st.markdown("---")
    st.header("Original Image")

    st.image(
        original,
        caption="Uploaded photograph",
        use_container_width=True
    )

    st.header("Reconstruction Results")

    metric_rows = []
    zip_buffer = BytesIO()

    with zipfile.ZipFile(
        zip_buffer, mode="w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        archive.writestr(
            "original.png",
            image_png_bytes(original)
        )

        for method, (degraded, restored) in experiments.items():
            st.markdown("---")
            st.subheader(method)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    degraded,
                    caption="Degraded Image",
                    use_container_width=True
                )

            with col2:
                st.image(
                    restored,
                    caption="Reconstructed Image",
                    use_container_width=True
                )

            with col3:
                st.image(
                    original,
                    caption="Original Reference",
                    use_container_width=True
                )

            mse, psnr, ssim = evaluate_image(
                original, restored
            )

            metric_rows.append({
                "Method": method,
                "MSE": round(mse, 2),
                "PSNR (dB)": round(psnr, 2),
                "SSIM": round(ssim, 4)
            })

            st.caption(
                "Metrics compare the reconstructed image "
                "with the original reference."
            )

            archive.writestr(
                method.replace(" ", "_").lower()
                + "_degraded.png",
                image_png_bytes(degraded)
            )

            archive.writestr(
                method.replace(" ", "_").lower()
                + "_reconstructed.png",
                image_png_bytes(restored)
            )

            st.download_button(
                f"Download {method} Result",
                data=image_png_bytes(restored),
                file_name=(
                    method.replace(" ", "_").lower()
                    + "_reconstructed.png"
                ),
                mime="image/png",
                key=f"download_{method}"
            )

        st.markdown("---")
        st.header("Quality Evaluation")

        results_df = pd.DataFrame(metric_rows)
        st.dataframe(results_df, use_container_width=True)

        st.caption(
            "Lower MSE and generally higher PSNR and SSIM "
            "indicate better fidelity to the original image. "
            "Results depend on the photograph and settings."
        )

        archive.writestr(
            "quality_metrics.csv",
            results_df.to_csv(index=False)
        )

    st.download_button(
        "Download All Results (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="image_reconstruction_results.zip",
        mime="application/zip",
        type="primary"
    )

    st.success("All three reconstruction experiments are complete!")
