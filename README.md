### Real-Time-Identity-Consistent-Face-Restoration

## Overview

This package provides a practical pipeline for face restoration on both images and videos.
It first detects one or multiple faces, applies a per-face SR model, and then pastes the enhanced results back into the original frame with careful blending—aiming for identity consistency, sharper details, natural colors, and stable results in real-world footage (meetings, compressed clips, group photos).

## What it does

![流程圖](/flow_chart.png)

**1. Tiled-grid video/image in → face/ROI detection.**

* The system ingests a low-resolution (LR) frame (e.g., a tiled robot-cam view), finds N faces using an MTCNN-style detector, and extracts per-face regions for later processing.

**2. Per-face quality estimation.**

* For each detected face we compute two simple quality cues:
    1. the face width in pixels (a proxy for available detail in the grid cell), and
    2. a sharpness score qvol based on the variance of the Laplacian (higher = sharper, lower = blurrier).

**3. Tiered routing by face size (w) and sharpness (qvol).**

* Faces are routed through size bands with a lightweight gate:
    * Large faces (≈ w ≥ 150 px): already recognizable → resize only (skip SR).
    * Medium faces (≈ 100–150 px): quality-aware mixture-of-experts:
        * if qvol < qlow (blurry/noisy) → Expert-T (transformer-leaning expert)
        * else → Expert-F (frequency/detail-leaning expert)
    * Small faces (≈ 50–100 px): route to Expert-T (robust to limited detail).
    * Very small faces (≈ w < 50 px): skip (not enough facial detail to enhance reliably). This “expert selection” follows the standard Mixture-of-Experts idea: different specialists are chosen per input based on a simple gate.

**4. Background upscale (separate from faces).**

* The non-face background is upscaled independently so global sharpness improves without over-processing faces or introducing halos.

**5. Seamless compositing back into the frame.**

* Enhanced faces are blended and pasted back onto the upscaled background using Poisson seamless cloning, which preserves illumination and edge continuity so the result looks natural.

## When to use it

* Meeting recordings, lecture captures, CCTV or heavily compressed social media clips where faces look soft or blocky.

* Group photos or multi-person frames where several faces need enhancement at once.

## How to use it
weights:[hybrid-transformer_model.pth.zip](./hybrid-transformer_model.pth.zip)

Image — face-only SR

Script: [photo_infer_faceonly_transformer.py](https://github.com/Krystalkao/Real-Time-Identity-Consistent-Face-Restoration/blob/821b5121921b314352a9962a79c5c4ab494fba54/photo_infer_faceonly_transformer.py)

Detects multiple faces in a still image (or a folder), runs ×4 SR per face, and saves LR/SR comparison tiles plus an overview mosaic.

Edit the parameters at the top of the file (input/output/weights), or run it with your CLI flags if you added them.

Outputs: per-face *_lr.png, *_sr.png, *_card.png and <image>_faces_overview.png.

Video — SR paste-back with blending

Script: [video_infer_pasteback_transformer.py](https://github.com/Krystalkao/Real-Time-Identity-Consistent-Face-Restoration/blob/821b5121921b314352a9962a79c5c4ab494fba54/video_infer_pasteback_transformer.py)

Per-frame detection → per-face ×4 SR → luminance-domain multi-band blending back to the full frame.

Produces three videos for easy inspection: baseline (scaled only), SR result, and side-by-side comparison.

Edit the parameters at the top of the file (video path/output/weights), or run it with your CLI flags if you added them.

![pic1](/compare_transformer_side_2ppl.png)
![pic2](/compare_transformer_side_yz.png)
![pic3](/compare_transformer_side_cc.png)

