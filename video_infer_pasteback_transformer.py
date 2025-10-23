#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, math, warnings, random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

# ===================== 使用者參數 =====================
VIDEO_PATH   = r"C:\Users\Krystal\Desktop\transformer_model\video_test\face_video\1m_R.avi"
OUT_DIR      = r"C:\Users\Krystal\Desktop\transformer_model\outputs_newmodel_refine\outputs_fullvideo_newmodel"
DEVICE       = "cuda"                      # 'cuda' or 'cpu'
TARGET_W, TARGET_H = 960, 2160            # 最終 4:9 尺寸（先把整幀拉到這個尺寸，再貼回 SR）
MIN_FACE_PX  = 24                          # 偵測最小臉短邊
FACE_SCORE_T = 0.50                        # 偵測分數門檻
UPSAMPLE     = 4                           # SR 倍數（訓練 ×4）

# 時序與幾何（影片用）
DETECT_EVERY        = 12                   # 每 N 幀強制偵測
BOX_TTL             = 12                   # 偵測失敗時沿用上一幀 box 的壽命
ROI_PAD_FRAC        = 0.06                 # ROI 外擴比例
QUANTIZE_PX         = 2                    # 中心/尺寸量化(px)

# 融合（分頻亮度融合）
PYR_LEVELS          = 5
KEEP_BASE_COARSE    = 2
MASK_INNER_FRAC     = 0.88
DETAIL_GAIN         = 1.25                 # 高頻細節增益

# 顯示/輸出
DRAW_DEBUG      = False
MAKE_COMPARISON = True
LABELS          = ("Original", "SR")
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 1.2
THICK           = 2

# 使用哪些模型
USE_PART_MODELS     = False                # 預設 False：只跑全臉模型
APPLY_Y_ONLY_MATCH  = True                 # 只調亮度避免色偏

# 權重
FACE_WEIGHTS  = r"C:\Users\Krystal\Desktop\transformer_model\weights_4models\face_base\latest.pth"
EYE_WEIGHTS   = r"C:\Users\Krystal\Desktop\transformer_model\weights_4models\eyes_matched\latest.pth"
NOSE_WEIGHTS  = r"C:\Users\Krystal\Desktop\transformer_model\weights_4models\nose_matched\latest.pth"
MOUTH_WEIGHTS = r"C:\Users\Krystal\Desktop\transformer_model\weights_4models\mouth_matched\latest.pth"

# ===================== 小工具 =====================
ToTensor, ToPIL = transforms.ToTensor(), transforms.ToPILImage()

def is_ok(p):
    if p is None: return True
    try:
        return (not np.isnan(float(p))) and (float(p) >= FACE_SCORE_T)
    except Exception:
        return True

def safe_clip_box(box, W, H):
    x1,y1,x2,y2 = [int(v) for v in box]
    x1 = max(0, min(x1, W-1)); y1 = max(0, min(y1, H-1))
    x2 = max(x1+1, min(int(x2), W)); y2 = max(y1+1, min(int(y2), H))
    return [x1,y1,x2,y2]

def quantize(v, q=2): return int(round(v / q) * q)

def create_writer(path_noext, fps, size_wh):
    trials = [('mp4v', '.mp4'), ('XVID', '.avi'), ('avc1', '.mp4')]
    for fourcc_str, ext in trials:
        p = str(Path(path_noext).with_suffix(ext))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        w,h = size_wh
        wr = cv2.VideoWriter(p, fourcc, fps, (w, h))
        if wr.isOpened(): return wr, p
    raise RuntimeError("無法建立VideoWriter，請檢查編碼器/副檔名/路徑")

def pad_box(box, W, H, frac=ROI_PAD_FRAC):
    x1,y1,x2,y2 = map(int, box)
    w = x2 - x1; h = y2 - y1
    pad = int(round(max(w,h)*frac))
    return safe_clip_box([x1-pad, y1-pad, x2+pad, y2+pad], W, H)

# ---------- 僅調亮度(Y)的微匹配，避免色偏 ----------
def y_only_match(patch_rgb, base_rgb):
    bycc = cv2.cvtColor(base_rgb,  cv2.COLOR_RGB2YCrCb).astype(np.float32)
    pycc = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    bY = bycc[...,0]; pY = pycc[...,0]
    pm, ps = pY.mean(), pY.std() + 1e-5
    bm, bs = bY.mean(), bY.std() + 1e-5
    pY_adj = np.clip((pY - pm) * (bs/ps) + bm, 0, 255)
    pycc[...,0] = pY_adj
    out = cv2.cvtColor(np.clip(pycc,0,255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return out

# ---------- 分頻亮度融合（只混高頻，不動色度） ----------
def cosine_ellipse_mask(h, w, inner_frac=0.88):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w-1)/2.0, (h-1)/2.0
    rx, ry = (w*inner_frac)/2.0, (h*inner_frac)/2.0
    nx = (xx - cx) / max(rx, 1e-6)
    ny = (yy - cy) / max(ry, 1e-6)
    r2 = nx*nx + ny*ny
    mask = (r2 <= 1.0).astype(np.float32)
    edge = (r2 > 1.0) & (r2 < 1.2)
    t = np.clip((r2-1.0)/0.2, 0, 1)
    mask[edge] = 0.5*(1.0 + np.cos(np.pi*t[edge]))
    mask = cv2.GaussianBlur(mask, (0,0), 0.02*min(h,w))
    return mask

def build_gauss_pyr(img, levels):
    pyr=[img]
    for _ in range(levels-1):
        img = cv2.pyrDown(img)
        pyr.append(img)
    return pyr

def build_lap_pyr(img, levels):
    gp = build_gauss_pyr(img, levels)
    lp=[]
    for i in range(levels-1):
        up = cv2.pyrUp(gp[i+1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(gp[i] - up)
    lp.append(gp[-1])
    return lp

def reconstruct_from_lap(lp):
    img = lp[-1]
    for i in range(len(lp)-2, -1, -1):
        img = cv2.pyrUp(img, dstsize=(lp[i].shape[1], lp[i].shape[0])) + lp[i]
    return img

def pyr_luma_blend_detail(base_rgb, patch_rgb, inner_frac=0.88, levels=5, keep_base_coarse=2, detail_gain=1.25):
    H,W = patch_rgb.shape[:2]
    bycc = cv2.cvtColor(base_rgb,  cv2.COLOR_RGB2YCrCb).astype(np.float32)
    pycc = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    bY, pY = bycc[...,0]/255.0, pycc[...,0]/255.0

    mask = cosine_ellipse_mask(H, W, inner_frac).astype(np.float32)
    gpM = build_gauss_pyr(mask, levels)

    lpB = build_lap_pyr(bY, levels)
    lpP = build_lap_pyr(pY, levels)

    blended=[]
    for i in range(levels):
        if i >= levels - keep_base_coarse:
            blended.append(lpB[i])
        else:
            Mi = gpM[i]
            detail = (lpP[i] - lpB[i]) * detail_gain   # 只加高頻差
            blended.append(lpB[i] + Mi * detail)

    outY = np.clip(reconstruct_from_lap(blended), 0.0, 1.0)
    out = bycc.copy()
    out[...,0] = (outY*255.0)
    out = cv2.cvtColor(np.clip(out,0,255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return out

# ===================== SRNet（與訓練同構） =====================
class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32, res_scale=0.2):
        super().__init__()
        self.res_scale=res_scale
        self.c1=nn.Conv2d(nf,gc,3,1,1)
        self.c2=nn.Conv2d(nf+gc,gc,3,1,1)
        self.c3=nn.Conv2d(nf+2*gc,nf,3,1,1)
        self.act=nn.LeakyReLU(0.2,True)
    def forward(self,x):
        x1=self.act(self.c1(x))
        x2=self.act(self.c2(torch.cat([x,x1],1)))
        x3=self.c3(torch.cat([x,x1,x2],1))
        return x + x3*self.res_scale

class SimpleSRTransformer(nn.Module):
    def __init__(self, in_ch=64, dim=64, depth=4, heads=4, ff=4, scale=4):
        super().__init__()
        self.conv_in  = nn.Conv2d(in_ch, dim, 3,1,1)
        enc = nn.TransformerEncoderLayer(dim, heads, dim*ff, batch_first=True, norm_first=True)
        self.encoder  = nn.TransformerEncoder(enc, depth)
        self.conv_pre = nn.Conv2d(dim, dim*(scale**2), 3,1,1)
        self.ps       = nn.PixelShuffle(scale)
        self.conv_out = nn.Conv2d(dim, 3, 3,1,1)
    def forward(self,x):
        x=self.conv_in(x)
        b,c,h,w=x.shape
        x=self.encoder(x.flatten(2).transpose(1,2))
        x=x.transpose(1,2).contiguous().view(b,c,h,w)
        return self.conv_out(self.ps(self.conv_pre(x)))

class SRNet(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3,64,3,1,1), nn.LeakyReLU(0.2,True))
        self.rrdb = nn.Sequential(RRDB(64,32,0.2), RRDB(64,32,0.2))
        self.body = SimpleSRTransformer(in_ch=64, dim=64, depth=4, heads=4, ff=4, scale=scale)
        self.detail = nn.Conv2d(3, 3, 1)
    def forward(self,x):
        x=self.head(x); x=self.rrdb(x); base=self.body(x)
        out = base + self.detail(base)
        return out, base

def load_G_from_ckpt(model, path):
    ck = torch.load(path, map_location="cpu")
    state = ck.get("G") or ck.get("model") or ck.get("state_dict") or ck
    if isinstance(state, dict):
        state = {k.replace('module.', ''): v for k,v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# ======== SR 推論包裝（支援 FP16） ========
@torch.inference_mode()
def infer_sr(model, pil_img, device, use_amp=True):
    x = ToTensor(pil_img).unsqueeze(0).to(device, non_blocking=True)  # [1,3,H,W], float32 0~1
    if use_amp and device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            y, _ = model(x)
    else:
        y, _ = model(x)
    y = y.squeeze(0).clamp(0,1).cpu()
    return ToPIL(y)

def composite_face_only(lr_face_pil, models, device, use_amp=True):
    sr_full = infer_sr(models["full"], lr_face_pil, device, use_amp=use_amp)
    return sr_full, {"eyes":0.0, "nose":0.0, "mouth":0.0}

# （如需開眼鼻口可再啟用）
def composite_full_then_parts(lr_face_pil, models, device, mtcnn_single, use_amp=True):
    full_m, eye_m, nose_m, mouth_m = models["full"], models["eye"], models["nose"], models["mouth"]
    sr_full = infer_sr(full_m, lr_face_pil, device, use_amp=use_amp)
    comp_np = np.array(sr_full).astype(np.uint8)
    try:
        box, lm5 = detect_once(lr_face_pil, mtcnn_single)
    except RuntimeError:
        return Image.fromarray(comp_np), {"eyes":0.0, "nose":0.0, "mouth":0.0}
    per_ms = {"eyes":0.0, "nose":0.0, "mouth":0.0}
    for region, m in [("eyes", eye_m), ("nose", nose_m), ("mouth", mouth_m)]:
        if m is None: continue
        patch_pil, rect = crop_region(lr_face_pil, box, lm5, region)
        if patch_pil is None: continue
        l, t, r, b = rect
        t0 = time.perf_counter()
        sr_patch = infer_sr(m, patch_pil, device, use_amp=use_amp)
        per_ms[region] = (time.perf_counter() - t0) * 1000.0
        patch_hr = np.array(sr_patch.resize(((r-l)*UPSAMPLE, (b-t)*UPSAMPLE), Image.LANCZOS)).astype(np.uint8)
        comp_roi = comp_np[t*UPSAMPLE:b*UPSAMPLE, l*UPSAMPLE:r*UPSAMPLE]
        fused = pyr_luma_blend_detail(comp_roi, patch_hr,
                                      inner_frac=MASK_INNER_FRAC,
                                      levels=PYR_LEVELS,
                                      keep_base_coarse=KEEP_BASE_COARSE,
                                      detail_gain=DETAIL_GAIN)
        comp_np[t*UPSAMPLE:b*UPSAMPLE, l*UPSAMPLE:r*UPSAMPLE] = fused
    return Image.fromarray(comp_np.astype(np.uint8)), per_ms

# ======== 五官/偵測幾何 ========
REGION_CFG = {
    "eyes":  dict(scale=0.50, padding=0.15),
    "nose":  dict(scale=0.35, padding=0.12),
    "mouth": dict(scale=0.45, padding=0.15),
}

def detect_once(pil_img, mtcnn_single):
    img = np.array(pil_img)
    boxes, probs, lms = mtcnn_single.detect(img, landmarks=True)
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("ROI 內偵測不到人臉/landmarks")
    i = int(np.argmax(probs))
    return boxes[i].astype(int), lms[i].astype(int)

def crop_region(pil_img, box, lm5, region):
    lm = np.array(lm5)
    if region == "eyes":   pts = lm[:2]
    elif region == "nose": pts = lm[2:3]
    else:                  pts = lm[3:]
    cx, cy = pts.mean(axis=0)
    face_w = (box[2] - box[0])
    cfg    = REGION_CFG[region]
    half   = face_w * cfg["scale"] / 2
    pad    = face_w * cfg["padding"]
    l = int(cx - half - pad); t = int(cy - half - pad)
    r = int(cx + half + pad); b = int(cy + half + pad)
    l, t = max(l,0), max(t,0)
    r, b = min(r,pil_img.width), min(b,pil_img.height)
    if r <= l or b <= t: return None, None
    return pil_img.crop((l,t,r,b)), (l,t,r,b)

# ===================== 影片主流程 =====================
def main():
    warnings.filterwarnings("ignore")

    # 性能優化：啟用 cudnn benchmark（卷積輸入固定時可更快）
    torch.backends.cudnn.benchmark = True  # 參考 PyTorch 文檔說明可自動選最佳算法
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"▶ DEVICE: {device.type}")

    # 模型
    full_m  = SRNet(scale=UPSAMPLE).to(device).eval()
    load_G_from_ckpt(full_m,  FACE_WEIGHTS)
    eye_m = nose_m = mouth_m = None
    if USE_PART_MODELS:
        eye_m   = SRNet(scale=UPSAMPLE).to(device).eval()
        nose_m  = SRNet(scale=UPSAMPLE).to(device).eval()
        mouth_m = SRNet(scale=UPSAMPLE).to(device).eval()
        load_G_from_ckpt(eye_m,   EYE_WEIGHTS)
        load_G_from_ckpt(nose_m,  NOSE_WEIGHTS)
        load_G_from_ckpt(mouth_m, MOUTH_WEIGHTS)
    models = {"full": full_m, "eye": eye_m, "nose": nose_m, "mouth": mouth_m}

    use_amp = (device.type == "cuda")  # 在 CUDA 上用 AMP

    # 影片 IO
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片：{VIDEO_PATH}"); return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    in_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = TARGET_W/float(in_w), TARGET_H/float(in_h)
    dt = 1.0 / max(1e-6, fps)
    stem  = Path(VIDEO_PATH).stem

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    writer_lr, path_lr = create_writer(Path(OUT_DIR, f"{stem}_lr_{TARGET_W}x{TARGET_H}").as_posix(), fps, (TARGET_W, TARGET_H))
    writer_sr, path_sr = create_writer(Path(OUT_DIR, f"{stem}_sr_{TARGET_W}x{TARGET_H}").as_posix(), fps, (TARGET_W, TARGET_H))
    print(f"✅ LR baseline → {path_lr}")
    print(f"✅ SR result   → {path_sr}")

    writer_cmp = None
    if MAKE_COMPARISON:
        cmp_size = (TARGET_W * 2, TARGET_H)
        writer_cmp, path_cmp = create_writer(Path(OUT_DIR, f"{stem}_cmp_{cmp_size[0]}x{cmp_size[1]}").as_posix(),
                                             fps, cmp_size)
        print(f"✅ Comparison  → {path_cmp}")

    # 偵測器：在 GPU 上跑；偵測用縮小圖（更快）
    det_device = device if device.type == "cuda" else "cpu"
    mtcnn_full = MTCNN(keep_all=False, device=det_device, select_largest=True, min_face_size=MIN_FACE_PX)
    mtcnn_roi  = MTCNN(keep_all=False, device=det_device) if USE_PART_MODELS else None
    DETECT_SHORT_SIDE = 640  # 縮到短邊 640 再偵測

    # One-Euro 濾波器（輕量實作）
    class LowPass:
        def __init__(self): self.y=None
        def reset(self): self.y=None
        def apply(self, x, alpha):
            if self.y is None: self.y=x
            self.y = self.y + alpha*(x - self.y)
            return self.y
    def alpha(dt, cutoff):
        tau = 1.0 / (2.0*math.pi*cutoff)
        return 1.0 / (1.0 + tau/dt)
    class OneEuro:
        def __init__(self, min_cutoff=1.2, beta=0.007, d_cutoff=1.0):
            self.min_cutoff=min_cutoff; self.beta=beta; self.d_cutoff=d_cutoff
            self.x_f=LowPass(); self.dx_f=LowPass(); self.last_x=None
        def reset(self):
            self.x_f.reset(); self.dx_f.reset(); self.last_x=None
        def apply(self, x, dt):
            if self.last_x is None: self.last_x = x
            dx = (x - self.last_x)/max(dt,1e-6)
            edx = self.dx_f.apply(dx, alpha(dt, self.d_cutoff))
            cutoff = self.min_cutoff + self.beta*abs(edx)
            self.last_x = x
            return self.x_f.apply(x, alpha(dt, cutoff))

    filt_cx = OneEuro(1.2,0.007,1.0)
    filt_cy = OneEuro(1.2,0.007,1.0)
    filt_w  = OneEuro(1.0,0.010,1.0)
    filt_h  = OneEuro(1.0,0.010,1.0)

    applied_count, skipped_count = 0, 0
    prev_box, prev_alive = None, 0
    frame_idx = 0
    need_detect = True

    t0_all = time.time()
    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        frame_idx += 1

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        base_rgb = cv2.resize(rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)

        # 偵測策略：縮小圖偵測 + 週期重偵測 + TTL
        use_box = None
        if need_detect or (frame_idx % DETECT_EVERY == 0) or prev_box is None:
            # 縮小到短邊 DETECT_SHORT_SIDE
            sh, sw = rgb.shape[:2]
            scale = DETECT_SHORT_SIDE / float(min(sh, sw))
            if scale < 1.0:
                det_img = cv2.resize(rgb, (int(round(sw*scale)), int(round(sh*scale))), interpolation=cv2.INTER_AREA)
                inv = 1.0/scale
            else:
                det_img = rgb; inv = 1.0
            try:
                boxes, probs = mtcnn_full.detect(det_img, landmarks=False)
            except Exception:
                boxes, probs = None, None
            if boxes is not None and len(boxes) > 0:
                j = int(np.argmax(probs)) if probs is not None else 0
                if is_ok(None if probs is None else probs[j]):
                    bx = [int(round(v*inv)) for v in boxes[j]]
                    use_box = safe_clip_box(bx, sw, sh)
                    prev_box, prev_alive = use_box, BOX_TTL
                    need_detect = False
                elif prev_box is not None and prev_alive > 0:
                    use_box, prev_alive = prev_box, prev_alive - 1
            elif prev_box is not None and prev_alive > 0:
                use_box, prev_alive = prev_box, prev_alive - 1
        else:
            if prev_box is not None and prev_alive > 0:
                use_box, prev_alive = prev_box, prev_alive - 1

        sr_frame = base_rgb.copy()
        if use_box is not None:
            x1,y1,x2,y2 = map(float, use_box)
            # 平滑 + 量化
            cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
            w  = max(2.0, x2-x1); h = max(2.0, y2-y1)
            cx_s = filt_cx.apply(cx, dt);  cy_s = filt_cy.apply(cy, dt)
            w_s  = filt_w.apply(w, dt);    h_s  = filt_h.apply(h, dt)
            w_s = quantize(w_s, QUANTIZE_PX); h_s = quantize(h_s, QUANTIZE_PX)
            cx_s = quantize(cx_s, QUANTIZE_PX); cy_s = quantize(cy_s, QUANTIZE_PX)
            x1,y1,x2,y2 = int(round(cx_s - w_s/2)), int(round(cy_s - h_s/2)), int(round(cx_s + w_s/2)), int(round(cy_s + h_s/2))
            smooth_box = pad_box([x1,y1,x2,y2], rgb.shape[1], rgb.shape[0], ROI_PAD_FRAC)
            prev_box = smooth_box

            x1,y1,x2,y2 = smooth_box
            roi_lr = rgb[y1:y2, x1:x2]
            if roi_lr.size > 0:
                roi_pil = Image.fromarray(roi_lr)
                if USE_PART_MODELS:
                    sr_face_pil, _ = composite_full_then_parts(roi_pil, models, device, mtcnn_roi, use_amp=use_amp)
                else:
                    sr_face_pil, _ = composite_face_only(roi_pil, models, device, use_amp=use_amp)

                # 轉到最終尺寸座標
                x1_f, y1_f = int(round(x1*sx)), int(round(y1*sy))
                w_lr, h_lr = (x2-x1), (y2-y1)
                patch_w = max(2, int(round(w_lr*sx)))
                patch_h = max(2, int(round(h_lr*sy)))
                sr_face_up = sr_face_pil.resize((patch_w, patch_h), Image.LANCZOS)
                sr_np  = np.array(sr_face_up).astype(np.uint8)

                base_roi = sr_frame[y1_f:y1_f+patch_h, x1_f:x1_f+patch_w]

                if APPLY_Y_ONLY_MATCH and base_roi.shape[:2] == sr_np.shape[:2]:
                    sr_np = y_only_match(sr_np, base_roi)

                fused = pyr_luma_blend_detail(base_roi, sr_np,
                                              inner_frac=MASK_INNER_FRAC,
                                              levels=PYR_LEVELS,
                                              keep_base_coarse=KEEP_BASE_COARSE,
                                              detail_gain=DETAIL_GAIN)
                sr_frame[y1_f:y1_f+patch_h, x1_f:x1_f+patch_w] = fused
                applied_count += 1
                if DRAW_DEBUG:
                    cv2.rectangle(sr_frame, (x1_f, y1_f), (x1_f+patch_w, y1_f+patch_h), (0,255,0), 2)
            else:
                skipped_count += 1
        else:
            # 偵測不到 → reset 濾波器以免漂移
            filt_cx.reset(); filt_cy.reset(); filt_w.reset(); filt_h.reset()
            skipped_count += 1

        # 輸出
        writer_lr.write(cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR))
        writer_sr.write(cv2.cvtColor(sr_frame,  cv2.COLOR_RGB2BGR))

        if MAKE_COMPARISON:
            cmp_rgb = np.empty((TARGET_H, TARGET_W * 2, 3), dtype=np.uint8)
            cmp_rgb[:, :TARGET_W] = base_rgb
            cmp_rgb[:, TARGET_W:] = sr_frame
            cmp_bgr = cv2.cvtColor(cmp_rgb, cv2.COLOR_RGB2BGR)
            cv2.line(cmp_bgr, (TARGET_W, 0), (TARGET_W, TARGET_H), (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(cmp_bgr, LABELS[0], (20, 60), FONT, FONT_SCALE, (255, 255, 255), THICK, cv2.LINE_AA)
            cv2.putText(cmp_bgr, LABELS[1], (TARGET_W + 20, 60), FONT, FONT_SCALE, (255, 255, 255), THICK, cv2.LINE_AA)
            writer_cmp.write(cmp_bgr)

        # 輕量進度
        if frame_idx % 50 == 0:
            elapsed = time.time() - t0_all
            print(f"進度 {frame_idx}/{total} | 成功貼回 {applied_count} 幀，略過 {skipped_count} 幀 | 用時 {elapsed:.1f}s")

    cap.release(); writer_lr.release(); writer_sr.release(); cv2.destroyAllWindows()
    if MAKE_COMPARISON and writer_cmp is not None:
        writer_cmp.release()

    total_frames = applied_count + skipped_count
    if total_frames > 0:
        pct = 100.0 * applied_count / total_frames
        print(f"✅ 完成：貼回成功 {applied_count}/{total_frames} 幀（{pct:.1f}%）")

if __name__ == "__main__":
    main()
