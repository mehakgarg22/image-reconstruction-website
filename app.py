    else:
        try:
            with st.spinner("Running restoration experiments..."):
                # A) Synthetic noise -> denoising. Keep noise input and denoising output distinct.
                rng = np.random.default_rng(42)
                noise = rng.normal(0, noise_strength, original.shape)
                noisy = np.clip(
                    original.astype(np.float32) + noise, 0, 255
                ).astype(np.uint8)
                denoised = denoise_image(noisy, denoise_strength)
                denoised = unsharp_mask(denoised, amount=sharpen_amount, radius=sharpen_radius) if sharpen_amount > 0 else denoised

                # B) Synthetic blur -> Richardson-Lucy deconvolution.
                blurred = cv2.GaussianBlur(
                    original, (int(blur_size), int(blur_size)), 0
                )
                deblurred = deblur_image(
                    blurred, iterations=deblur_iterations,
                    kernel_size=int(blur_size), sigma=blur_sigma
                )
                deblurred = unsharp_mask(
                    deblurred, amount=sharpen_amount * 0.35,
                    radius=sharpen_radius
                ) if sharpen_amount > 0 else deblurred

                # C) Missing patch -> inpainting. The patch is simulated at center.
                damaged = original.copy()
                mask = np.zeros((height, width), dtype=np.uint8)
                patch_w = max(1, int(width * missing_percent / 100))
                patch_h = max(1, int(height * missing_percent / 100))
                x1 = (width - patch_w) // 2
                y1 = (height - patch_h) // 2
                x2, y2 = x1 + patch_w, y1 + patch_h
                damaged[y1:y2, x1:x2] = 0
                mask[y1:y2, x1:x2] = 255
                inpainted = cv2.inpaint(
                    damaged, mask, 3, cv2.INPAINT_TELEA
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
                metrics = compare_metric_rows(original, restored)

            st.markdown('<div class="section-head">Original vs. reconstructed</div>', unsafe_allow_html=True)
            st.caption("Compare each reconstruction with its degraded input and original reference. Use the slider settings to tune the output.")
            method_choice = st.selectbox(
                "Choose experiment to inspect",
                list(restored.keys()),
                key="method_choice"
            )
            dcol, rcol = st.columns(2, gap="large")
            with dcol:
                st.markdown("**Degraded input**")
                st.image(degraded[method_choice], use_container_width=True)
            with rcol:
                st.markdown("**Reconstructed output**")
                st.image(restored[method_choice], use_container_width=True)

            st.markdown('<div class="section-head">Quality evaluation</div>', unsafe_allow_html=True)
            st.caption("Metrics compare each reconstructed image with the uploaded original. They are reference-based and may not perfectly reflect perceived quality.")
            st.dataframe(metrics, use_container_width=True, hide_index=True)

            selected_row = metrics[metrics["Experiment"] == method_choice].iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("MSE", f'{selected_row["MSE ↓"]:.3f}', help="Lower is generally better.")
            m2.metric("PSNR", f'{selected_row["PSNR (dB) ↑"]:.2f} dB', help="Higher is generally better.")
            m3.metric("SSIM", f'{selected_row["SSIM ↑"]:.4f}', help="Closer to 1 generally indicates greater structural similarity.")

            st.markdown('<div class="section-head">Download your results</div>', unsafe_allow_html=True)
            download_cols = st.columns(3)
            method_names = list(restored.keys())
            for i, name in enumerate(method_names):
                with download_cols[i]:
                    st.download_button(
                        f"Download {name}",
                        data=to_png_bytes(restored[name]),
                        file_name=name.lower().replace(" ", "_") + "_reconstructed.png",
                        mime="image/png",
                        key=f"individual_{i}",
                        use_container_width=True
                    )

            zip_data = build_results_zip(original, degraded, restored, metrics)
            st.download_button(
                "Download complete report (ZIP)",
                data=zip_data,
                file_name="imagerestore_results.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
            st.success("Experiments completed. If a result looks over-smoothed, lower denoising strength; if edges look harsh, lower detail enhancement.")
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            st.caption("Try a smaller image or lower the deblurring iteration count.")

with guide_tab:
    st.markdown("### Restoration methods")
    st.markdown(
        """
        - **Noise removal:** Non-local means denoising reduces simulated random noise. Strong denoising may soften fine textures.
        - **Blur restoration:** Richardson–Lucy deconvolution estimates a sharper image from a known/estimated blur kernel. It works best for mild blur that resembles the assumed kernel and can produce ringing or noise when the estimate is wrong.
        - **Missing-region repair:** OpenCV Telea inpainting estimates pixels in a simulated rectangular damaged region from nearby pixels. It cannot recover unknown original details exactly.
        """
    )
    st.markdown("### Metric interpretation")
    st.markdown(
        """
        - **MSE:** mean squared pixel error; lower is better.
        - **PSNR:** signal-to-noise ratio relative to the reference; higher is better.
        - **SSIM:** structural similarity; values closer to 1 generally indicate greater similarity.

        Because the app creates the degraded input from the uploaded original, it has a reference image for evaluation. For a real damaged photo with no original reference, these metrics cannot be calculated in the same way.
        """
    )
    st.warning(
        "The app does not guarantee that every restoration will improve an already-clean photo. "
        "Use low denoising strength for clean images and compare outputs before downloading."
    )
