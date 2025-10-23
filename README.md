### Real-Time-Identity-Consistent-Face-Restoration

## Overview

This package provides a practical pipeline for face restoration on both images and videos. 

It first detects one or multiple faces, applies an SR model to each face, then pastes the enhanced results back into the original frame with careful blending—aiming for identity consistency, sharper details, natural colors, and stable results in real-world footage (meetings, compressed clips, group photos).

## What it does

**Tiled-grid video/image in → face/ROI detection.**

* The system ingests a low-resolution (LR) frame (e.g., a tiled robot-cam view), finds N faces using an MTCNN-style detector, and extracts per-face regions for later processing.

**Per-face quality estimation.**

* For each detected face we compute two simple quality cues:
    1. the face width in pixels (a proxy for available detail in the grid cell), and
    2. a sharpness score qvol based on the variance of the Laplacian (higher = sharper, lower = blurrier).

**Tiered routing by face size (w) and sharpness (qvol).**

* Faces are routed through size bands with a lightweight gate:
    * Large faces (≈ w ≥ 150 px): already recognizable → resize only (skip SR).
    * Medium faces (≈ 100–150 px): quality-aware mixture-of-experts:
        * if qvol < qlow (blurry/noisy) → Expert-T (transformer-leaning expert)
        * else → Expert-F (frequency/detail-leaning expert)
    * Small faces (≈ 50–100 px): route to Expert-T (robust to limited detail).
    * Very small faces (≈ w < 50 px): skip (not enough facial detail to enhance reliably). This “expert selection” follows the standard Mixture-of-Experts idea: different specialists are chosen per input based on a simple gate.

**Background upscale (separate from faces).**

* The non-face background is upscaled independently so global sharpness improves without over-processing faces or introducing halos.

**Seamless compositing back into the frame.**

* Enhanced faces are blended and pasted back onto the upscaled background using Poisson seamless cloning, which preserves illumination and edge continuity so the result looks natural.

## When to use it

* Meeting recordings, lecture captures, CCTV or heavily compressed social media clips where faces look soft or blocky.

* Group photos or multi-person frames where several faces need enhancement at once.
!(compare_transformer_side_2ppl.png)
!()
