# 🖼️ ImageRestore Studio

**An Image Restoration Web Application Built with Python and Streamlit**

ImageRestore Studio is a web-based application that helps improve the visual quality of photographs using image-processing techniques. Users can upload an image, adjust restoration settings, and automatically compare the original image with the processed result.

## ✨ Features

- **Image Upload:** Upload JPG, JPEG, PNG, and WEBP images.
- **Automatic Processing:** Image processing starts automatically after uploading an image or changing settings.
- **Noise Removal:** Reduce visible noise using OpenCV denoising techniques.
- **Scratch and Speck Repair:** Attempt to detect and repair certain scratches and small marks automatically.
- **Contrast Enhancement:** Adjust image contrast to improve visibility.
- **Sharpening:** Enhance edges and make details appear clearer.
- **Side-by-Side Comparison:** View the original and reconstructed images together.
- **Quality Metrics:** Calculate Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM).
- **Download Results:** Download the original image, reconstructed image, and a ZIP archive containing results and a processing report.
- **Resolution Preservation:** Maintain the uploaded image's dimensions during processing.

## 🛠️ Technologies Used

- **Python** — Core programming language
- **Streamlit** — Web application interface
- **OpenCV** — Image processing, denoising, and inpainting
- **NumPy** — Numerical and array operations
- **Pillow (PIL)** — Image loading and conversion
- **scikit-image** — Image quality metrics

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/image-reconstruction-website.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Navigate to the project directory

```bash
cd image-reconstruction-website
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🚀 How to Use

1. Open ImageRestore Studio.
2. Upload a photograph using the image uploader.
3. Adjust denoising strength, contrast, sharpening, and damage-repair settings if needed.
4. Wait for the application to process the image automatically.
5. Compare the original photograph with the reconstructed result.
6. Review the MSE, PSNR, and SSIM metrics.
7. Download the reconstructed image or all results as a ZIP file.

## 📊 Image Quality Metrics

| Metric | Description |
|---|---|
| MSE | Measures the average squared pixel difference between two images. |
| PSNR | Expresses pixel-level reconstruction error on a logarithmic scale. |
| SSIM | Estimates structural similarity between two images. |

**Note:** These metrics compare the processed image with the uploaded image. Since the undamaged reference image is generally unavailable, the values do not independently establish how accurately missing details have been recovered.

## 📁 Project Structure

```text
image-reconstruction-website/
│
├── app.py
├── requirements.txt
└── README.md
```

## ☁️ Deployment

The application can be deployed using [Streamlit Community Cloud](https://share.streamlit.io/).

1. Push `app.py`, `requirements.txt`, and `README.md` to your GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Create a new app and connect your repository.
4. Select `app.py` as the main file.
5. Deploy the application.

## ⚠️ Limitations

- Automatic scratch detection is heuristic-based and may mistakenly identify real image textures as damage.
- Traditional image-processing techniques cannot reliably reconstruct completely missing objects, facial features, or large image regions.
- Sharpening and denoising can alter fine details if used too aggressively.
- Results depend on the quality and type of the uploaded image.

## 🔮 Future Improvements

- Integration of AI-based image inpainting models.
- More accurate automatic damage detection.
- Restoration of larger missing image regions.
- Before-and-after comparison slider.
- Additional image restoration algorithms.

## 👩‍💻 Author

**Mehak Garg**

- GitHub: https://github.com/mehakgarg22
- LinkedIn:www.linkedin.com/in/mehak-garg-1a1850321

---

⭐ If you find this project useful, consider giving the repository a star!
