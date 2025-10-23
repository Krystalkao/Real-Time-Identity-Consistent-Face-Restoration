#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
infer_multiface_srnet.py
多臉匡臉 → 以你訓練腳本的 SRNet(×4) 權重做超解析
- 單檔可跑：內建 SRNet/RRDB/SimpleSRTransformer/DetailHead 定義
- 使用 facenet-pytorch 的 MTCNN 進行多臉偵測
- 自動 pad 到 ×4 整除、推論後再 unpad
- 逐臉輸出 LR/SR 與小卡；另存總覽圖
"""

import os, time, math, warnings
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt

# ===================== 參數 =====================
IMAGE_PATH   = r"C:\Users\Krystal\Desktop\transformer_model\fortest\multifaces.jpg"   # ← 你的輸入圖片
WEIGHTS_FACE = r"C:\Users\Krystal\Desktop\transformer_model\weights_4models\face_base\latest.pth"  # ← 你的訓練權重
OUT_DIR      = r"C:\Users\Krystal\Desktop\transformer_model\outputs_newmodel_refine"                   # ← 輸出資料夾
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
UPSCALE      = 4
DET_THR      = 0.80
SAVE_CARD    = True

# Warm-up 次數（避免冷啟動 timing 偏高）
WARMUP_FORWARD = 3
WARMUP_E2E     = 1

torch.backends.cudnn.benchmark = True
ToTensor, ToPIL = transforms.ToTensor(), transforms.ToPILImage()

# ===================== 與訓練腳本一致的模型 =====================
class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.c1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.c2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.c3 = nn.Conv2d(nf + 2 * gc, nf, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], 1)))
        x3 = self.c3(torch.cat([x, x1, x2], 1))
        return x + x3 * self.res_scale

class SimpleSRTransformer(nn.Module):
    def __init__(self, in_ch=64, dim=64, depth=4, heads=4, ff=4, scale=4):
        super().__init__()
        self.conv_in  = nn.Conv2d(in_ch, dim, 3, 1, 1)
        enc = nn.TransformerEncoderLayer(dim, heads, dim * ff, batch_first=True, norm_first=True)
        self.encoder  = nn.TransformerEncoder(enc, depth)
        self.conv_pre = nn.Conv2d(dim, dim * (scale ** 2), 3, 1, 1)
        self.ps       = nn.PixelShuffle(scale)
        self.conv_out = nn.Conv2d(dim, 3, 3, 1, 1)
    def forward(self, x):
        x = self.conv_in(x)
        b, c, h, w = x.shape
        x = self.encoder(x.flatten(2).transpose(1, 2))
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        return self.conv_out(self.ps(self.conv_pre(x)))

class DetailHead(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32)
        self.register_buffer('hp', k.view(1,1,3,3).repeat(3,1,1,1))
        self.gate = nn.Sequential(nn.Conv2d(3, 3, 1), nn.Tanh())
    def forward(self, base):
        hp = F.conv2d(base, self.hp, padding=1, groups=3)
        g  = self.gate(hp) * 0.3
        return base + g * hp

class SRNet(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.rrdb = nn.Sequential(RRDB(64, 32, 0.2), RRDB(64, 32, 0.2))
        self.body = SimpleSRTransformer(in_ch=64, dim=64, depth=4, heads=4, ff=4, scale=scale)
        self.detail_head = DetailHead()
    def forward(self, x):
        x = self.head(x); x = self.rrdb(x); base = self.body(x)
        out = self.detail_head(base)
        return out, base

# ===================== 工具函式 =====================
def load_generator(weights_path: str, device: torch.device) -> nn.Module:
    G = SRNet(scale=UPSCALE).to(device).eval()
    ck = torch.load(weights_path, map_location="cpu")
    state = None
    if isinstance(ck, dict):
        # 優先抓訓練腳本存的 key
        for k in ["G", "model", "state_dict"]:
            if k in ck and isinstance(ck[k], dict):
                state = ck[k]; break
    if state is None:
        state = ck
    missing = G.load_state_dict(state, strict=False)  # 容忍非關鍵鍵值
    # 你若想看有哪些鍵沒對上，可印出 missing
    return G

def pad_to_multiple(img: torch.Tensor, m: int = 4):
    """img: 1x3xHxW → 反射 padding 到 H,W 為 m 的倍數；回傳 padded, (padT,padB,padL,padR)"""
    _, _, H, W = img.shape
    pad_h = (m - (H % m)) % m
    pad_w = (m - (W % m)) % m
    pt = pad_h // 2; pb = pad_h - pt
    pl = pad_w // 2; pr = pad_w - pl
    if pad_h or pad_w:
      img = F.pad(img, (pl, pr, pt, pb), mode="reflect")
    return img, (pt, pb, pl, pr)

def unpad(img: torch.Tensor, pads):
    pt, pb, pl, pr = pads
    if (pt+pb+pl+pr) == 0: return img
    return img[..., pt: img.shape[-2]-pb if pb>0 else None,
               pl: img.shape[-1]-pr if pr>0 else None]

@torch.inference_mode()
def infer_one_face(G: nn.Module, roi_pil: Image.Image, device: torch.device):
    """單臉 ROI → SRNet 推論（含 ×4 整除 padding 與 CUDA Event 計時）"""
    x = ToTensor(roi_pil).unsqueeze(0).to(device)
    x, pads = pad_to_multiple(x, m=UPSCALE)

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        y, _ = G(x)
        ender.record()
        torch.cuda.synchronize()
        ms = starter.elapsed_time(ender)  # 毫秒
    else:
        t0 = time.perf_counter()
        y, _ = G(x)
        ms = (time.perf_counter() - t0) * 1000.0

    y = y.clamp(0,1)
    y = unpad(y, pads)
    return ToPIL(y.squeeze(0).cpu()), ms

def warmup(G: nn.Module, roi_pil: Image.Image, device: torch.device, n=3):
    with torch.inference_mode():
        x = ToTensor(roi_pil).unsqueeze(0).to(device)
        x, _ = pad_to_multiple(x, m=UPSCALE)
        for _ in range(max(0, n)):
            _ = G(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

# ===================== 主流程 =====================
def main():
    warnings.filterwarnings("ignore")
    device = torch.device(DEVICE)
    print(f"▶ DEVICE: {device.type}")

    # 1) 載入大圖
    big = Image.open(IMAGE_PATH).convert("RGB")
    big_np = np.array(big)

    # 2) 多臉偵測
    mtcnn = MTCNN(keep_all=True, device="cpu")
    t_det0 = time.perf_counter()
    boxes, probs, _ = mtcnn.detect(big_np, landmarks=True)  # landmarks=True 可同時取回關鍵點（本版先用框）
    t_det_ms = (time.perf_counter() - t_det0) * 1000.0

    faces = []
    if boxes is not None:
        H, W = big_np.shape[:2]
        for b, p in zip(boxes, probs):
            if p is None or p < DET_THR: continue
            x1,y1,x2,y2 = map(int, b)
            x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H))
            if (x2-x1) >= 2 and (y2-y1) >= 2:
                faces.append((x1,y1,x2,y2))

    if not faces:
        print("未偵測到人臉"); return

    # 3) 載入 SRNet 權重
    print("載入 SRNet 權重…")
    G = load_generator(WEIGHTS_FACE, device)

    # 4) Warm-up（用第一張臉避免冷啟動）
    faces.sort(key=lambda b: ((b[1]+b[3])//2, (b[0]+b[2])//2))  # 先上後下、左到右
    roi0 = Image.fromarray(big_np[faces[0][1]:faces[0][3], faces[0][0]:faces[0][2]])
    warmup(G, roi0, device, n=WARMUP_FORWARD)
    for _ in range(WARMUP_E2E):
        _ = infer_one_face(G, roi0, device)

    # 5) 建輸出目錄
    stem = Path(IMAGE_PATH).stem
    out_dir = Path(OUT_DIR) / f"{stem}_faces"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6) 逐臉處理
    print(f"🔍 多臉偵測：{t_det_ms:.2f} ms | 臉數={len(faces)}")
    cards = []
    for i, (x1,y1,x2,y2) in enumerate(faces, 1):
        roi_lr = Image.fromarray(big_np[y1:y2, x1:x2])
        lr_path = out_dir / f"face_{i:02d}_lr.png"
        roi_lr.save(lr_path)

        t0 = time.perf_counter()
        sr_img, t_forward_ms = infer_one_face(G, roi_lr, device)
        t_e2e_ms = (time.perf_counter() - t0) * 1000.0
        sr_path = out_dir / f"face_{i:02d}_sr.png"
        sr_img.save(sr_path)

        if SAVE_CARD:
            Wf, Hf = sr_img.size
            lr_up = roi_lr.resize((Wf, Hf), Image.BICUBIC)
            card = Image.new("RGB", (Wf*2, Hf))
            card.paste(lr_up, (0,0)); card.paste(sr_img, (Wf,0))
            card.save(out_dir / f"face_{i:02d}_card.png")
            cards.append(np.array(card))

        print(f"[Face {i:02d}] box=({x1},{y1},{x2},{y2}) → "
              f"Full(MO)={t_forward_ms:.2f} ms | E2E={t_e2e_ms:.2f} ms | "
              f"保存：{lr_path.name}, {sr_path.name}")

    # 7) 總覽輸出
    if SAVE_CARD and cards:
        cols = 2
        rows = math.ceil(len(cards)/cols)
        plt.figure(figsize=(6*cols, 3*rows))
        for i, p in enumerate(cards, 1):
            plt.subplot(rows, cols, i)
            plt.imshow(p); plt.axis("off")
            plt.title(f"Face {i}: LR↑ | SR")
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_faces_overview.png", dpi=200)
        plt.close()

    print(f"✅ 完成：輸出在 {out_dir}")

if __name__ == "__main__":
    main()
