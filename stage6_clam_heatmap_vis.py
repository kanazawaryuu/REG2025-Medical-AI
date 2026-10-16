# stage6_clam_heatmap_vis.py
# (方案 B 终极版: 复刻官方 CLAM 高密重叠采样 + A_raw 基准分位数校准 + 三语论文大图与单图套件)
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import glob
import time
import urllib.request
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image, PngImagePlugin
from scipy.stats import rankdata
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import openslide

PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 新增：定义 CLAM 官方蓝紫红专属色阶（中间以浅紫过渡，杜绝纯白冲淡组织）
from matplotlib.colors import LinearSegmentedColormap
CLAM_BLUERED = LinearSegmentedColormap.from_list(
    'clam_bluered',
    [(0.00, '#264e86'),   # 低注意: 经典冷靛蓝
     (0.35, '#5d76aa'),   # 次低: 浅蓝紫
     (0.50, '#9a729e'),   # 中间过渡: 灰紫（关键：跳过纯白，边界锐利清楚！）
     (0.68, '#c95965'),   # 次高: 珊瑚粉红
     (1.00, '#b51825')]   # 高注意: 鲜红
)

# =====================================================================
# 0. 多语言字典 (EN / JA / ZH) 与 Linux 免权限中日文字体引擎
# =====================================================================
I18N = {
    'en': {
        'he_title': 'H&E Original',
        'ens_name': 'Ensemble (Ours)',
        'cbar_label': 'Attention Weight (Ref Percentile)',
        'fig1_title': 'Cross-Model Dense Attention Comparison Across Breast Biopsy Subtypes',
        'fig2_title': 'Multi-Branch Spatial Disentanglement on Complex Multi-Lesion Biopsy',
        'fig3_title': 'Comparison of Pure Dense Map (Nature Style) vs. Histology Overlay (CLAM Official)',
        'row_Pure': 'Pure Dense Map (Jet)',
        'row_Overlay': 'CLAM Official Overlay',
        'label_invasive_nst': 'Invasive NST',
        'label_dcis': 'DCIS (In Situ)',
        'label_invasive_lobular': 'Invasive Lobular',
        'label_mucinous': 'Mucinous Ca.',
        'label_microcalcification': 'Microcalcification',
        'label_fibroadenoma': 'Fibroadenoma',
        'label_papillary_neoplasm': 'Papillary Neoplasm',
        'label_no_tumor': 'No Tumor',
    },
    'ja': {
        'he_title': 'H&E 原画像',
        'ens_name': 'アンサンブル (提案手法)',
        'cbar_label': '注意重み (基準パーセンタイル)',
        'fig1_title': '乳腺生検サブタイプにおける高密度Attentionマップのモデル間比較',
        'fig2_title': '複合病変切片におけるマルチブランチ空間的注意分離',
        'fig3_title': '高密度純粋マップ (Nature様式) と組織重畳マップ (CLAM公式様式) の比較',
        'row_Pure': '高密度純粋マップ (Jet)',
        'row_Overlay': 'CLAM公式 組織重畳マップ',
        'label_invasive_nst': '浸潤性乳管癌 (NST)',
        'label_dcis': '非浸潤性乳管癌 (DCIS)',
        'label_invasive_lobular': '浸潤性小葉癌 (ILC)',
        'label_mucinous': '粘液癌',
        'label_microcalcification': '微小石灰化',
        'label_fibroadenoma': '線維腺腫',
        'label_papillary_neoplasm': '乳頭状腫瘍',
        'label_no_tumor': '腫瘍なし',
    },
    'zh': {
        'he_title': 'H&E 原始切片',
        'ens_name': '多模集成 (本文方案)',
        'cbar_label': '注意力权重 (基准分位数校准)',
        'fig1_title': '不同乳腺穿刺病理亚型下的高密度跨模型注意力分布对比',
        'fig2_title': '复杂多病灶穿刺切片的多分支空间注意力解耦可视化',
        'fig3_title': '顶刊纯净热力风格 (Nature) 与官方组织叠加风格 (CLAM) 对比',
        'row_Pure': '纯净高密模式 (Jet)',
        'row_Overlay': 'CLAM 官方组织叠加',
        'label_invasive_nst': '非特殊型浸润癌 (NST)',
        'label_dcis': '导管原位癌 (DCIS)',
        'label_invasive_lobular': '浸润性小叶癌 (ILC)',
        'label_mucinous': '黏液癌',
        'label_microcalcification': '微钙化灶',
        'label_fibroadenoma': '纤维腺瘤',
        'label_papillary_neoplasm': '乳头状肿瘤',
        'label_no_tumor': '未见肿瘤',
    }
}

def setup_cjk_font():
    cjk_candidates = [
        'Noto Sans CJK JP', 'Noto Sans CJK SC', 'Noto Sans CJK TC',
        'Source Han Sans JP', 'Source Han Sans SC', 'Source Han Sans CN',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'IPAGothic', 'TakaoGothic',
        'SimHei', 'Microsoft YaHei', 'Arial Unicode MS'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in cjk_candidates:
        if cand in available:
            print(f"🔤 [字体引擎] 已启用系统内置中日文字体: {cand}")
            return cand

    font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.fonts')
    os.makedirs(font_dir, exist_ok=True)
    font_path = os.path.join(font_dir, 'NotoSansSC-Regular.otf')
    if not os.path.exists(font_path):
        url = "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
        try:
            print("🌐 [字体引擎] 正在下载开源中日文字体 (NotoSansCJKjp-Regular.otf)...")
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            return 'DejaVu Sans'
    try:
        fm.fontManager.addfont(font_path)
        return fm.FontProperties(fname=font_path).get_name()
    except Exception:
        return 'DejaVu Sans'

# =====================================================================
# 1. Stage 1 高密重叠滑动窗口特征提取引擎 (复用 Stage 1 架构 + 本地缓存)
# =====================================================================
def _worker_init_fn(worker_id):
    if cv2 is not None:
        cv2.setNumThreads(0)

class FastRAMTileDataset(Dataset):
    """
    【核武器级加速】抛弃 5 万次 OpenSlide 重复磁盘解压！
    一次性将组织有效包围盒 (Bounding Box) 读入 RAM 内存 (仅占数百 MB)，
    后续 5 万个重叠窗口全部为纯内存 O(1) 零拷贝数组切片，读图耗时从 9 分钟降至 0.5 秒！
    """
    def __init__(self, wsi_path: str, coords: np.ndarray, region_size: int, tile_size: int):
        self.coords = coords
        self.region_size = region_size
        self.tile_size = tile_size

        # 计算所有有效瓦片的最小外接矩形 (Bounding Box)
        self.x0 = max(0, int(coords[:, 0].min()))
        self.y0 = max(0, int(coords[:, 1].min()))
        x_max = int(coords[:, 0].max()) + region_size
        y_max = int(coords[:, 1].max()) + region_size

        with openslide.OpenSlide(str(wsi_path)) as slide:
            w_lim, h_lim = slide.dimensions
            box_w = min(x_max, w_lim) - self.x0
            box_h = min(y_max, h_lim) - self.y0
            # 分条带安全读入内存画布 (仅解压 1 次 TIFF！)
            self.canvas = np.full((box_h, box_w, 3), 255, dtype=np.uint8)
            strip_h = 4096
            for sy in range(0, box_h, strip_h):
                cur_h = min(strip_h, box_h - sy)
                region_pil = slide.read_region((self.x0, self.y0 + sy), 0, (box_w, cur_h)).convert("RGB")
                self.canvas[sy:sy + cur_h, :] = np.asarray(region_pil, dtype=np.uint8)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        rx = int(self.coords[idx, 0]) - self.x0
        ry = int(self.coords[idx, 1]) - self.y0
        patch = self.canvas[ry:ry + self.region_size, rx:rx + self.region_size]
        if patch.shape[0] != self.region_size or patch.shape[1] != self.region_size:
            pad_p = np.full((self.region_size, self.region_size, 3), 255, dtype=np.uint8)
            pad_p[:patch.shape[0], :patch.shape[1]] = patch
            patch = pad_p
        if self.region_size != self.tile_size:
            patch = cv2.resize(patch, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR)
        return patch

def load_foundation_backbone(backbone_key: str):
    """加载视觉基座，返回 (model, mean_rgb, std_rgb, target_input_size, region_size, tile_size)"""
    print(f"\n🚀 [Stage 1 引擎] 正在加载视觉基座大模型: {backbone_key} ...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if backbone_key == 'H-optimus-1':
        model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5, dynamic_img_size=False)
        model.to(DEVICE).eval()
        return model, (0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517), 224, 224, 224

    elif backbone_key == 'Virchow2':
        model = timm.create_model("hf_hub:paige-ai/Virchow2", pretrained=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
        model.to(DEVICE).eval()
        return model, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224, 224, 224

    elif backbone_key == 'CONCH1.5':
        weights_path = "/home/zjin/weights/conch1_5/pytorch_model_vision.bin"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"未找到 CONCH 1.5 权重文件: {weights_path}")

        print(f"  ⚡ [本地直读] 正在加载 CONCH 1.5 官方真实纯视觉权重: {weights_path}")
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=448,
            patch_size=16,
            init_values=1.0,
            num_classes=0,
            dynamic_img_size=True
        )
        state_dict = torch.load(weights_path, map_location="cpu")
        cleaned_sd = {k.replace("trunk.", ""): v for k, v in state_dict.items() if k.startswith("trunk.")}
        load_info = model.load_state_dict(cleaned_sd if cleaned_sd else state_dict, strict=False)
        print(f"  ✅ CONCH 1.5 权重载入完成 (匹配项: {len(cleaned_sd)})")
        
        model.to(DEVICE).eval()
        return model, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 448, 224, 224

    elif backbone_key == 'GigaPath':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        model.to(DEVICE).eval()
        return model, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224, 448, 224

    raise ValueError(f"Unknown backbone: {backbone_key}")

def generate_dense_overlapping_coords(wsi_path: str, ref_coords: np.ndarray, patch_size: int = 224, stride: int = 112) -> np.ndarray:
    """自适应步长保护：若单切片原始瓦片 > 4500，自动微调步长防止单图突破 6 万瓦片"""
    eff_stride = max(stride, int(stride * np.sqrt(len(ref_coords) / 3500.0))) if len(ref_coords) > 4500 else stride
    with openslide.OpenSlide(str(wsi_path)) as slide:
        w0, h0 = slide.dimensions
        scale = max(w0, h0) / 2048.0
        tw, th = max(int(w0 / scale), 100), max(int(h0 / scale), 100)
        thumb = np.array(slide.get_thumbnail((tw, th)).convert("RGB"))
        th, tw = thumb.shape[:2]

    support_mask = np.zeros((th, tw), dtype=np.uint8)
    sx, sy = tw / w0, th / h0
    pw_t, ph_t = max(int(np.ceil(patch_size * sx)), 2), max(int(np.ceil(patch_size * sy)), 2)
    for x, y in ref_coords:
        cx, cy = int(x * sx), int(y * sy)
        support_mask[max(0, cy - ph_t // 4):min(th, cy + ph_t + ph_t // 4),
                     max(0, cx - pw_t // 4):min(tw, cx + pw_t + pw_t // 4)] = 1

    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    tissue_px = ((hsv[:, :, 1] >= 12) | (gray < 228)) & (gray > 10)
    valid_mask = (support_mask > 0) & tissue_px

    x_min = max(0, int(ref_coords[:, 0].min()) - eff_stride)
    x_max = min(w0 - patch_size, int(ref_coords[:, 0].max()) + eff_stride)
    y_min = max(0, int(ref_coords[:, 1].min()) - eff_stride)
    y_max = min(h0 - patch_size, int(ref_coords[:, 1].max()) + eff_stride)

    dense_list = []
    for y in range(y_min, y_max + 1, eff_stride):
        ty1, ty2 = int(y * sy), min(th, int((y + patch_size) * sy))
        for x in range(x_min, x_max + 1, eff_stride):
            tx1, tx2 = int(x * sx), min(tw, int((x + patch_size) * sx))
            sub = valid_mask[ty1:ty2, tx1:tx2]
            if sub.size > 0 and (np.count_nonzero(sub) / sub.size) >= 0.12:
                dense_list.append([x, y])

    return np.array(dense_list, dtype=np.int32)

@torch.inference_mode()
def extract_dense_features_with_backbone(backbone_key: str, model, mean_rgb, std_rgb, target_input_sz: int,
                                         wsi_path: str, dense_coords: np.ndarray, region_size: int, tile_size: int,
                                         batch_size: int = 512) -> np.ndarray:
    """基于 FastRAMTileDataset 内存零 I/O 极速提取"""
    t_load = time.time()
    dataset = FastRAMTileDataset(wsi_path, dense_coords, region_size, tile_size)
    print(f"    ⚡ [RAM 预载完成] 整片组织区域已一次性载入内存 ({dataset.canvas.shape[1]}x{dataset.canvas.shape[0]} px, 耗时 {time.time()-t_load:.1f}s)")

    # 数据已在内存，4 个轻量 worker 即可跑满 PCIe / NVLink 带宽
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, prefetch_factor=4, shuffle=False, pin_memory=True)
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    mean_t = torch.tensor(mean_rgb, device=DEVICE, dtype=amp_dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std_rgb, device=DEVICE, dtype=amp_dtype).view(1, 3, 1, 1)
    all_feats = []

    for batch_uint8 in tqdm(loader, desc=f"  [RAM->GPU 极速推理 {backbone_key}]", leave=False):
        x = batch_uint8.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).to(amp_dtype).div_(255.0)
        if target_input_sz != tile_size:
            x = F.interpolate(x, size=(target_input_sz, target_input_sz), mode='bicubic', align_corners=False)
        x = x.sub_(mean_t).div_(std_t)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            out = model(x)
            feats = out[:, 0] if backbone_key == 'Virchow2' else (out[0] if isinstance(out, (tuple, list)) else out)
        all_feats.append(feats.float().cpu().numpy())

    del dataset
    return np.vstack(all_feats)        

# =====================================================================
# 2. Stage 3/4 CLAM-MB 模型定义与 A_raw 基准分位数校准器
# =====================================================================
class PredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class MultiBranchAttentionMIL(nn.Module):
    def __init__(self, n_classes, input_dim=1280, hidden_dim=512, dropout=0.25, idx_nst=0, idx_dcis=1):
        super().__init__()
        self.n_classes = n_classes
        self.idx_nst = idx_nst
        self.idx_dcis = idx_dcis
        self.feature_extractor = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, 256), nn.Sigmoid())
        self.attention_weights = nn.Linear(256, n_classes)
        self.classifiers = nn.Linear(hidden_dim, n_classes)
        self.head_nst_tubule = PredictionHead(hidden_dim, 3)
        self.head_nst_nuclear = PredictionHead(hidden_dim, 3)
        self.head_nst_mitoses = PredictionHead(hidden_dim, 3)
        self.head_nst_overall = PredictionHead(hidden_dim, 3)
        self.head_dcis_grade = PredictionHead(hidden_dim, 3)
        self.head_dcis_necrosis = PredictionHead(hidden_dim, 3)
        self.head_dcis_types = PredictionHead(hidden_dim, 3)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(0)
        H = self.feature_extractor(x)
        A_raw = torch.transpose(self.attention_weights(self.attention_V(H) * self.attention_U(H)), 1, 0)
        A = F.softmax(A_raw, dim=1)
        M = torch.mm(A, H)
        logits_main = (M * self.classifiers.weight).sum(dim=1) + self.classifiers.bias
        return {'main': logits_main.unsqueeze(0)}, A, A_raw

@torch.no_grad()
def compute_calibrated_dense_attention(model_dir: str, ref_feat_path: str, dense_feats: np.ndarray,
                                       input_dim: int, n_classes: int, idx_nst: int, idx_dcis: int):
    """
    【官方 CLAM 核心机制】
    1. 用原始 3332 特征 (ref_feat) 算出真实预测概率 prob 与基准打分分布 A_raw_ref；
    2. 用高密特征 (dense_feats) 算出未归一化打分 A_raw_dense；
    3. 将 A_raw_dense 投影到 A_raw_ref 的百分位分布上 (use_ref_scores=True)，彻底杜绝 Softmax 失真！
    """
    ref_feats = np.load(ref_feat_path).astype(np.float32)
    x_ref = torch.from_numpy(ref_feats).float().to(DEVICE)
    x_dense = torch.from_numpy(dense_feats.astype(np.float32)).float().to(DEVICE)

    n_dense = x_dense.shape[0]
    n_ref = x_ref.shape[0]
    ckpt_dir = os.path.join(model_dir, 'checkpoints')

    dense_pct_sum = np.zeros((n_classes, n_dense), dtype=np.float32)
    ref_pct_sum = np.zeros((n_classes, n_ref), dtype=np.float32)
    prob_sum = np.zeros(n_classes, dtype=np.float32)
    valid_folds = 0

    for fold in range(5):
        p = os.path.join(ckpt_dir, f'best_model_fold{fold}.pth')
        if not os.path.exists(p):
            continue
        model = MultiBranchAttentionMIL(n_classes=n_classes, input_dim=input_dim, idx_nst=idx_nst, idx_dcis=idx_dcis).to(DEVICE)
        try:
            model.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
        except Exception:
            model.load_state_dict(torch.load(p, map_location=DEVICE))
        model.eval()

        # 1) 原始参考特征前向传播 -> 锁定真实预测概率与基准 A_raw_ref
        out_ref, _, A_raw_ref = model(x_ref)
        prob_sum += torch.sigmoid(out_ref['main']).cpu().numpy()[0]
        raw_ref_np = A_raw_ref.cpu().numpy()

        # 2) 高密重叠特征前向传播 -> 获取每个微窗口的独立 A_raw_dense
        _, _, A_raw_dense = model(x_dense)
        raw_dense_np = A_raw_dense.cpu().numpy()

        # 3) 向量化基准分位数校准 (score2percentile with reference distribution)
        for c_i in range(n_classes):
            ref_sorted = np.sort(raw_ref_np[c_i])
            pct_d = np.searchsorted(ref_sorted, raw_dense_np[c_i], side='right') / float(n_ref)
            dense_pct_sum[c_i] += pct_d
            ref_pct_sum[c_i] += rankdata(raw_ref_np[c_i]) / float(n_ref)

        valid_folds += 1
        del model

    torch.cuda.empty_cache()
    if valid_folds == 0:
        return None, None, None
    return dense_pct_sum / valid_folds, ref_pct_sum / valid_folds, prob_sum / valid_folds

# =====================================================================
# 3. 顶刊级高分辨率二维重叠投票画布渲染引擎 (Nature 风格 & CLAM 官方风格)
# =====================================================================
def load_highres_wsi_thumbnail(wsi_path: str, max_size: int = 3200):
    """加载高清 WSI 缩略图 (3200px 级)，保留清晰的腺管腔隙与细胞核群落纹理"""
    mpp = 0.25
    with openslide.OpenSlide(str(wsi_path)) as slide:
        w0, h0 = slide.dimensions
        if openslide.PROPERTY_NAME_MPP_X in slide.properties:
            mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        scale = min(max_size / w0, max_size / h0, 1.0)
        tw, th = int(w0 * scale), int(h0 * scale)
        thumb = np.array(slide.get_thumbnail((tw, th)).convert("RGB"))
        return thumb, (thumb.shape[1] / w0, thumb.shape[0] / h0), mpp

def compute_crop_box(coords, scale_xy, canvas_shape, patch_size=224, pad=30):
    h, w = canvas_shape[:2]
    sx, sy = scale_xy
    xs, ys = coords[:, 0] * sx, coords[:, 1] * sy
    pw, ph = max(int(patch_size * sx), 4), max(int(patch_size * sy), 4)
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + pw + pad), min(h, int(ys.max()) + ph + pad))

def prepare_clean_he(thumb_rgb, coords, scale_xy, patch_size=224, crop_box=None, mpp=0.25, draw_scale_bar=True):
    h, w = thumb_rgb.shape[:2]
    sx, sy = scale_xy
    pw, ph = max(int(patch_size * sx), 4), max(int(patch_size * sy), 4)

    tissue_mask = np.zeros((h, w), dtype=np.uint8)
    for cx, cy in np.column_stack([coords[:, 0] * sx, coords[:, 1] * sy]).astype(int):
        tissue_mask[max(0, cy):min(h, cy + ph), max(0, cx):min(w, cx + pw)] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(pw // 2, 5), max(ph // 2, 5)))
    tissue_mask = cv2.dilate(tissue_mask, kernel, iterations=1)

    clean_he = np.full_like(thumb_rgb, 255)
    valid = (tissue_mask > 0) & (thumb_rgb.mean(axis=2) < 242)
    clean_he[valid] = thumb_rgb[valid]

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        clean_he = clean_he[y1:y2, x1:x2]

    if draw_scale_bar:
        ch, cw = clean_he.shape[:2]
        px_per_mm = (1000.0 / max(mpp, 0.05)) * sx
        bar_mm = 5 if (px_per_mm * 5) < (cw * 0.28) else (2 if (px_per_mm * 2) < (cw * 0.28) else 1)
        bar_px = max(int(px_per_mm * bar_mm), 24)
        bar_h = max(int(ch * 0.016), 5)
        mx, my = int(cw * 0.04), int(ch * 0.06)
        y_top = ch - my - bar_h
        clean_he[y_top:y_top + bar_h, mx:mx + bar_px] = [15, 15, 15]
        return clean_he, f"{bar_mm} mm", (mx + bar_px / 2, y_top + bar_h)

    return clean_he, None, None

def accumulate_2d_canvas(pct_1d: np.ndarray, coords: np.ndarray, scale_xy: Tuple[float, float],
                         thumb_rgb: np.ndarray, patch_size: int = 224, stride: int = 112):
    h, w = thumb_rgb.shape[:2]
    sx, sy = scale_xy
    pw = max(int(np.ceil(patch_size * sx)), 2)
    ph = max(int(np.ceil(patch_size * sy)), 2)

    att_canvas = np.zeros((h, w), dtype=np.float32)
    weight_canvas = np.zeros((h, w), dtype=np.float32)

    norm_c = np.column_stack([coords[:, 0] * sx, coords[:, 1] * sy]).astype(int)
    for (cx, cy), val in zip(norm_c, pct_1d):
        x1, y1 = max(0, cx), max(0, cy)
        x2, y2 = min(w, cx + pw), min(h, cy + ph)
        if x2 > x1 and y2 > y1:
            att_canvas[y1:y2, x1:x2] += float(val)
            weight_canvas[y1:y2, x1:x2] += 1.0

    # 1. 最近邻特征值无缝外延 + 高斯平滑（消除一切方块锯齿与 0.0 蓝色断层）
    has_tile = weight_canvas > 0
    raw_avg = np.zeros_like(att_canvas)
    raw_avg[has_tile] = att_canvas[has_tile] / weight_canvas[has_tile]

    from scipy.ndimage import distance_transform_edt
    _, nearest_idx = distance_transform_edt(~has_tile, return_indices=True)
    filled_att = raw_avg[tuple(nearest_idx)]

    ksize = 51
    final_att = cv2.GaussianBlur(filled_att, (ksize, ksize), 0)

    # 2. 像素级严丝合缝组织掩码：只填补细胞间微隙，绝不在边缘向外鼓出圆包！
    weight_smooth = cv2.GaussianBlur(weight_canvas, (ksize, ksize), 0)
    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2GRAY)

    # 严格识别有染色像素，且必须在有效瓦片覆盖区内部
    raw_tissue = ((hsv[:, :, 1] >= 12) | (gray < 234)) & (gray > 15) & (weight_smooth > 0.10)

    # 仅用 5x5 极小核连接相邻细胞核，随后滤除面積小于 60 像素的孤立玻片灰尘
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue_mask = cv2.morphologyEx(raw_tissue.astype(np.uint8), cv2.MORPH_CLOSE, k_small)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tissue_mask, connectivity=8)
    clean_mask = np.zeros_like(tissue_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 60:
            clean_mask[labels == i] = 1

    return np.clip(final_att, 0.0, 1.0), (clean_mask > 0)

def render_nature_dense_map(avg_map: np.ndarray, mask: np.ndarray, thumb_rgb: np.ndarray, crop_box=None, alpha: float = 0.75):
    """
    【100% 还原 CLAM blank_canvas 渲染】
    纯白底板 + JET 色阶 + 纯白线性混合 + 连续平滑组织掩码
    """
    h, w = thumb_rgb.shape[:2]
    # 换回旧版的经典无白色彩虹色阶（蓝 -> 青 -> 绿 -> 黄 -> 橙 -> 红）
    cmap = plt.get_cmap('jet')
    heat_rgb = (cmap(avg_map)[:, :, :3] * 255.0).astype(np.float32)

    # 官方标准透明度混白：使红蓝两端清透不脏
    blended = alpha * heat_rgb + (1.0 - alpha) * 255.0

    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[mask] = np.clip(blended[mask], 0, 255).astype(np.uint8)

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        canvas = canvas[y1:y2, x1:x2]
    return canvas

def render_clam_official_overlay(avg_map: np.ndarray, mask: np.ndarray, clean_he_full: np.ndarray,
                                 crop_box=None, alpha: float = 0.42, cmap='jet'):
    """
    【100% 还原 CLAM 官方 visHeatmap(blank_canvas=False)】
    高清 H&E 组织底图 + 连续注意力场平滑线性叠加
    支持 'jet' 与 'coolwarm'，并对 coolwarm 自动执行中段去白对比度拉伸
    """
    cmap_name = cmap if isinstance(cmap, str) else getattr(cmap, 'name', '')
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    # 如果是 coolwarm / bwr，做微对比度拉伸（压缩 0.5 附近的白色过渡带，让红蓝交界更紧凑扎实）
    if 'coolwarm' in cmap_name or 'bwr' in cmap_name:
        contrast_factor =20.0  # 控制中间白色过渡宽度：数值越大，中间发白过渡越窄，红蓝交界越扎实聚拢
        map_to_render = np.clip((avg_map - 0.5) * contrast_factor + 0.5, 0.0, 1.0)
    else:
        map_to_render = avg_map

    heat_rgb = (cmap_obj(map_to_render)[:, :, :3] * 255.0).astype(np.float32)

    overlay = clean_he_full.copy()
    # 官方 cv2.addWeighted 融合公式：(1 - alpha) * H&E + alpha * Heatmap
    blended = (1.0 - alpha) * clean_he_full.astype(np.float32) + alpha * heat_rgb
    overlay[mask] = np.clip(blended[mask], 0, 255).astype(np.uint8)

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        overlay = overlay[y1:y2, x1:x2]
    return overlay

def _get_clam_tissue_hole_mask(thumb_rgb: np.ndarray, patch_mask: np.ndarray) -> np.ndarray:
    """
    【CLAM 官方 segmentTissue + use_holes=True 核心机制】
    提取真实组织轮廓，并将导管内腔孔洞、脂肪空泡与玻片背景严格抠除留白
    """
    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2GRAY)
    
    # 判定真实组织实质 (有苏木精/伊红染色且非纯白背景与黑边)
    tissue_with_holes = ((hsv[:, :, 1] >= 12) | (gray < 232)) & (gray > 12)
    
    # 与瓦片有效区域取交集
    mask = patch_mask & tissue_with_holes
    return mask


# =====================================================================
# 4. 模型间注意力相似度量化分析
# =====================================================================
def compute_attention_similarity_matrix(att_dict, top_k_ratio=0.10):
    names = list(att_dict.keys())
    n = len(names)
    lengths = [len(v) for v in att_dict.values()]
    target_len = max(set(lengths), key=lengths.count)

    aligned = {}
    for k, v in att_dict.items():
        r = rankdata(v) / len(v)
        if len(r) != target_len:
            r = np.interp(np.linspace(0.0, 1.0, target_len), np.linspace(0.0, 1.0, len(r)), r)
        aligned[k] = r

    cos_mat, iou_mat = np.eye(n), np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a1, a2 = aligned[names[i]], aligned[names[j]]
            cos_sim = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-8)
            cos_mat[i, j] = cos_mat[j, i] = round(float(cos_sim), 4)
            k_num = max(int(target_len * top_k_ratio), 1)
            t1, t2 = set(np.argsort(a1)[-k_num:]), set(np.argsort(a2)[-k_num:])
            iou_mat[i, j] = iou_mat[j, i] = round(float(len(t1 & t2) / max(len(t1 | t2), 1)), 4)

    return pd.DataFrame(cos_mat, index=names, columns=names), pd.DataFrame(iou_mat, index=names, columns=names)

# =====================================================================
# 5. 主程序：先按模型批量提取/读取高密缓存，再生成三语论文图集与全套单图
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Stage 6 Scheme B: Official CLAM Dense Overlapping Heatmap Suite")
    parser.add_argument('--csv_path', type=str, default='final_train_list_multilabel.csv')
    parser.add_argument('--slide_ids', type=str, nargs='+',
                        default=['PIT_01_00019_01', 'PIT_01_01794_01', 'PIT_01_00548_01', 'PIT_01_00105_01'])
    parser.add_argument('--stride', type=int, default=56,
                        help="密集滑动步长 (默认 56 即 75%% 重叠，比 224 细 16 倍；若设为 32 则细 49 倍)")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--target_class', type=str, default='auto')
    parser.add_argument('--wsi_dir', type=str, default='/home/zjin/wsi_dataset')
    parser.add_argument('--conch1_5_dir', type=str, default='train_stage3_conch15_clammb')
    parser.add_argument('--virchow2_dir', type=str, default='train_stage3_virchow2_clammb')
    parser.add_argument('--hoptimus_dir', type=str, default='train_stage3_hop1_clammb')
    parser.add_argument('--gigapath_dir', type=str, default='train_stage3_gigapath_clammb')
    parser.add_argument('--output_dir', type=str, default='evaluation_stage6_heatmaps')
    parser.add_argument('--langs', type=str, nargs='+', default=['en', 'ja', 'zh'])
    args = parser.parse_args()

    cjk_font = setup_cjk_font()
    matplotlib.rcParams['font.sans-serif'] = [cjk_font, 'DejaVu Sans', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False

    os.makedirs(args.output_dir, exist_ok=True)
    dense_cache_root = os.path.join(args.output_dir, f'dense_cache_s{args.stride}')
    os.makedirs(dense_cache_root, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    label_cols = [c for c in df.columns if c.startswith('label_')]
    idx_nst = label_cols.index('label_invasive_nst') if 'label_invasive_nst' in label_cols else 0
    idx_dcis = label_cols.index('label_dcis') if 'label_dcis' in label_cols else 1
    df['clean_id'] = df['wsi_id' if 'wsi_id' in df.columns else 'id'].astype(str).apply(lambda s: Path(s).stem)

    backbones = [
        ('CONCH1.5', args.conch1_5_dir, 'conch1_5', 1024),
        ('Virchow2', args.virchow2_dir, 'virchow2', 1280),
        ('H-optimus-1', args.hoptimus_dir, 'h_optimus_1', 1536),
        ('GigaPath', args.gigapath_dir, 'gigapath', 1536)
    ]

    # 1. 预先为每张目标切片生成统一的密集重叠坐标网格 (保证所有模型在完全相同的微窗口坐标上对齐)
    slide_rows = {}
    slide_wsi_paths = {}
    slide_ref_coords = {}
    slide_dense_coords = {}

    for sid in args.slide_ids:
        clean_sid = Path(sid).stem
        sub_df = df[df['clean_id'] == clean_sid]
        if sub_df.empty:
            print(f"⚠️ 未在 CSV 中找到切片 {clean_sid}，跳过！")
            continue
        row = sub_df.iloc[0]
        wsi_p = os.path.join(args.wsi_dir, f"{clean_sid}.tiff")
        if not os.path.exists(wsi_p):
            cand = glob.glob(os.path.join(args.wsi_dir, f"{clean_sid}.*"))
            wsi_p = cand[0] if cand else None
        if not wsi_p or not os.path.exists(wsi_p):
            print(f"⚠️ 未找到原始切片文件 {clean_sid}，跳过！")
            continue

        ref_cp = str(row['coord_path'])
        ref_c = np.load(ref_cp)[:, :2]
        dense_cp = os.path.join(dense_cache_root, f"{clean_sid}_dense_coords_s{args.stride}.npy")
        if os.path.exists(dense_cp):
            dense_c = np.load(dense_cp)
        else:
            dense_c = generate_dense_overlapping_coords(wsi_p, ref_c, patch_size=224, stride=args.stride)
            np.save(dense_cp, dense_c)

        slide_rows[clean_sid] = row
        slide_wsi_paths[clean_sid] = wsi_p
        slide_ref_coords[clean_sid] = ref_c
        slide_dense_coords[clean_sid] = dense_c
        print(f"📍 切片 [{clean_sid}] 坐标就绪: 原始无重叠瓦片={len(ref_c)} -> 高密重叠瓦片(stride={args.stride})={len(dense_c)}")

    # 2. 按模型依次加载基座网络提取高密特征
    for m_name, m_dir, col_prefix, dim in backbones:
        if not os.path.isdir(m_dir):
            continue
        m_cache_dir = os.path.join(dense_cache_root, m_name)
        os.makedirs(m_cache_dir, exist_ok=True)

        missing_sids = [sid for sid in slide_rows if not os.path.exists(os.path.join(m_cache_dir, f"{sid}_dense_feats.npy"))]
        if not missing_sids:
            print(f"⚡ [{m_name}] 全部目标切片的高密特征缓存已命中，跳过 GPU 基座推理！")
            continue

        try:
            backbone_model, mean_rgb, std_rgb, target_sz, region_sz, tile_sz = load_foundation_backbone(m_name)
            for sid in missing_sids:
                t0 = time.time()
                d_feats = extract_dense_features_with_backbone(
                    m_name, backbone_model, mean_rgb, std_rgb, target_sz, slide_wsi_paths[sid],
                    slide_dense_coords[sid], region_sz, tile_sz, batch_size=max(args.batch_size, 512)
                )
                np.save(os.path.join(m_cache_dir, f"{sid}_dense_feats.npy"), d_feats)
                print(f"  ✅ [{m_name}] 切片 {sid} 高密特征提取完成! 形状: {d_feats.shape} | 耗时: {time.time()-t0:.1f}s")
            del backbone_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ [{m_name}] 跳过在线高密重提 ({e})，将在第 3 步使用已有特征执行最近邻高密映射。")
            torch.cuda.empty_cache()

    # 3. 逐切片执行 CLAM A_raw 基准分位数校准与二维重叠投票画布渲染
    slides_cache = []
    from scipy.spatial import cKDTree

    for clean_sid, row in slide_rows.items():
        print(f"\n================ 正在合成切片 [{clean_sid}] 的微米级注意力热力图 ================")
        wsi_p = slide_wsi_paths[clean_sid]
        dense_c = slide_dense_coords[clean_sid]
        ref_c = slide_ref_coords[clean_sid]

        dense_atts, ref_atts, model_probs = {}, {}, {}

        for m_name, m_dir, col_prefix, dim in backbones:
            if not os.path.isdir(m_dir):
                continue
            feat_col = next((c for c in [f"{col_prefix}_feature_path", f"{col_prefix.replace('_', '-')}_feature_path"] if c in row), None)
            if not feat_col or not os.path.exists(str(row[feat_col])):
                continue
            ref_fp = str(row[feat_col])
            dense_fp = os.path.join(dense_cache_root, m_name, f"{clean_sid}_dense_feats.npy")

            if os.path.exists(dense_fp):
                d_feats = np.load(dense_fp)
            else:
                d_feats = np.load(ref_fp)
                orig_c_path = ref_fp.replace('_features_', '_coords_')
                orig_c = np.load(orig_c_path)[:, :2] if os.path.exists(orig_c_path) else ref_c
                from scipy.spatial import cKDTree
                _, nn_idx = cKDTree(orig_c).query(dense_c, k=1)
                d_feats = orig_f[nn_idx] if 'orig_f' in locals() else d_feats[nn_idx]

            d_pct, r_pct, prob_vec = compute_calibrated_dense_attention(
                m_dir, ref_fp, d_feats, dim, len(label_cols), idx_nst, idx_dcis
            )
            if d_pct is not None:
                dense_atts[m_name] = (d_pct, dense_c, 224 if m_name != 'GigaPath' else 448)
                ref_atts[m_name] = r_pct
                model_probs[m_name] = prob_vec
                print(f"  🎯 [{m_name:<11}] 完成 5 折 A_raw 基准分位数校准 (采样点: {d_pct.shape[1]})")

        # 构造冠军三模 (CONCH1.5 + Virchow2 + H-optimus-1) 高密融合注意力
        champ_keys = [k for k in ['CONCH1.5', 'Virchow2', 'H-optimus-1'] if k in dense_atts]
        if len(champ_keys) >= 2:
            aligned_pcts = []
            for k in champ_keys:
                p_arr, c_arr, _ = dense_atts[k]
                if len(c_arr) != len(dense_c):
                    _, idx_map = cKDTree(c_arr).query(dense_c, k=1)
                    aligned_pcts.append(p_arr[:, idx_map])
                else:
                    aligned_pcts.append(p_arr)
            ens_d_pct = np.mean(aligned_pcts, axis=0)
            ens_r_pct = np.mean([ref_atts[k] for k in champ_keys], axis=0)
            for c_i in range(len(label_cols)):
                ref_sorted = np.sort(ens_r_pct[c_i])
                ens_d_pct[c_i] = np.searchsorted(ref_sorted, ens_d_pct[c_i], side='right') / float(ens_r_pct.shape[1])
            dense_atts['Ensemble'] = (ens_d_pct, dense_c, 224)
            ref_atts['Ensemble'] = ens_r_pct
            model_probs['Ensemble'] = np.mean([model_probs[k] for k in champ_keys], axis=0)

        # 确定目标对比主诊断类别
        if args.target_class != 'auto' and args.target_class in label_cols:
            target_cls = args.target_class
        else:
            pos_cols = [c for c in label_cols if row.get(c, 0) >= 0.5 and c != 'label_no_tumor']
            target_cls = pos_cols[0] if pos_cols else 'label_invasive_nst'
        c_idx = label_cols.index(target_cls)

        # 导出量化相似度 CSV
        single_cls_dense = {k: dense_atts[k][0][c_idx] for k in dense_atts}
        df_cos, df_iou = compute_attention_similarity_matrix(single_cls_dense)
        df_cos.to_csv(os.path.join(args.output_dir, f"{clean_sid}_{target_cls}_cosine.csv"))
        df_iou.to_csv(os.path.join(args.output_dir, f"{clean_sid}_{target_cls}_top10_iou.csv"))

        # 加载 3200px 级高分辨率原图缩略图与裁剪框
        thumb_rgb, scale_xy, mpp = load_highres_wsi_thumbnail(wsi_p, max_size=3200)
        crop_box = compute_crop_box(ref_c, scale_xy, thumb_rgb.shape, patch_size=224)
        clean_he_crop, bar_text, bar_pos = prepare_clean_he(
            thumb_rgb, ref_c, scale_xy, patch_size=224, crop_box=crop_box, mpp=mpp, draw_scale_bar=True
        )
        clean_he_full, _, _ = prepare_clean_he(
            thumb_rgb, ref_c, scale_xy, patch_size=224, crop_box=None, mpp=mpp, draw_scale_bar=False
        )

        # 同时保留 Jet 经典彩虹与 BlueRed 官方蓝紫红
        clam_maps, clam_maps_bwr, probs_main_cls = {}, {}, {}
        for m_name, (d_mat, m_coords, m_psize) in dense_atts.items():
            avg_map, mask = accumulate_2d_canvas(d_mat[c_idx], m_coords, scale_xy, thumb_rgb, patch_size=m_psize)
            clam_maps[m_name] = render_clam_official_overlay(avg_map, mask, clean_he_full, crop_box=crop_box, cmap='jet')
            clam_maps_bwr[m_name] = render_clam_official_overlay(avg_map, mask, clean_he_full, crop_box=crop_box, cmap='coolwarm')
            probs_main_cls[m_name] = float(model_probs[m_name][c_idx])

        # 多分支空间解耦：也同时生成两套
        active_gt_cols = [c for c in label_cols if row.get(c, 0) >= 0.5]
        branch_renders = {}
        if len(active_gt_cols) >= 2 and 'Ensemble' in dense_atts:
            ens_d_mat, ens_coords, ens_ps = dense_atts['Ensemble']
            ens_p = model_probs['Ensemble']
            for b_col in active_gt_cols:
                b_idx = label_cols.index(b_col)
                b_avg, b_mask = accumulate_2d_canvas(ens_d_mat[b_idx], ens_coords, scale_xy, thumb_rgb, patch_size=ens_ps)
                branch_renders[b_col] = {
                    'overlay': render_clam_official_overlay(b_avg, b_mask, clean_he_full, crop_box=crop_box, cmap='jet'),
                    'overlay_bwr': render_clam_official_overlay(b_avg, b_mask, clean_he_full, crop_box=crop_box, cmap='coolwarm'),
                    'prob': float(ens_p[b_idx])
                }

        slides_cache.append({
            'sid': clean_sid,
            'target_cls': target_cls,
            'he_img': clean_he_crop,
            'bar_text': bar_text,
            'bar_pos': bar_pos,
            'clam_maps': clam_maps,
            'clam_maps_bwr': clam_maps_bwr,
            'probs': probs_main_cls,
            'branches': branch_renders
        })

    if not slides_cache:
        return

    # 4. 按语言版本 (EN / JA / ZH) 批量导出论文级组图 + 全套独立单图
    for lang in args.langs:
        if lang not in I18N:
            continue
        tr = I18N[lang]
        lang_dir = os.path.join(args.output_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)
        print(f"\n🎨 正在渲染 [{lang.upper()}] 语言版本的论文级组图与单图 -> {lang_dir}/")

        n_rows = len(slides_cache)
        model_order = [k for k in ['CONCH1.5', 'Virchow2', 'H-optimus-1', 'GigaPath', 'Ensemble'] if k in slides_cache[0]['clam_maps']]
        n_cols = 1 + len(model_order)

        # 【论文图 1】跨模型矩阵总图 (Paper_Fig1_CrossModel_Matrix)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols + 0.6, 3.3 * n_rows), dpi=350, facecolor='white')
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for r_i, s_data in enumerate(slides_cache):
            cls_label = tr.get(s_data['target_cls'], s_data['target_cls'].replace('label_', ''))
            ax0 = axes[r_i, 0]
            ax0.imshow(s_data['he_img'])
            ax0.set_xticks([]); ax0.set_yticks([])
            for sp in ax0.spines.values(): sp.set_visible(False)
            if s_data['bar_text'] and s_data['bar_pos']:
                bx, by = s_data['bar_pos']
                ax0.text(bx, by + s_data['he_img'].shape[0] * 0.045, s_data['bar_text'],
                         ha='center', va='top', fontsize=8.5, fontweight='bold', color='#111111')
            ax0.set_xlabel(f"{cls_label}\n({s_data['sid']})", fontsize=10.5, fontweight='bold', labelpad=6)

            for c_i, m_key in enumerate(model_order, start=1):
                ax = axes[r_i, c_i]
                ax.imshow(s_data['clam_maps'][m_key])
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_visible(False)
                disp_name = tr['ens_name'] if m_key == 'Ensemble' else m_key
                ax.set_xlabel(f"{disp_name}: {s_data['probs'][m_key]:.2f}", fontsize=10.5, fontweight='bold', labelpad=6)

        cbar_ax = fig.add_axes([0.96, 0.18, 0.012, 0.65])
        sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap='jet')
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(tr['cbar_label'], fontsize=10, fontweight='bold')

        plt.suptitle(tr['fig1_title'], fontsize=13.5, fontweight='bold', y=0.99)
        plt.subplots_adjust(wspace=0.06, hspace=0.26, right=0.94)
        for ext in ['png', 'pdf']:
            fig.savefig(os.path.join(lang_dir, f"Paper_Fig1_CrossModel_Matrix_{lang}.{ext}"), bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # 【论文图 2 & 3 及独立单图】
        for s_data in slides_cache:
            sid = s_data['sid']
            if s_data['branches']:
                b_cols = list(s_data['branches'].keys())
                nb = len(b_cols) + 1
                fig2, axes2 = plt.subplots(1, nb, figsize=(3.6 * nb, 3.8), dpi=350, facecolor='white')
                axes2[0].imshow(s_data['he_img'])
                axes2[0].set_xticks([]); axes2[0].set_yticks([])
                for sp in axes2[0].spines.values(): sp.set_visible(False)
                axes2[0].set_xlabel(f"{tr['he_title']}\n({sid})", fontsize=10.5, fontweight='bold')

                for b_i, b_col in enumerate(b_cols, start=1):
                    ax_b = axes2[b_i]
                    ax_b.imshow(s_data['branches'][b_col]['overlay'])
                    ax_b.set_xticks([]); ax_b.set_yticks([])
                    for sp in ax_b.spines.values(): sp.set_visible(False)
                    b_title = tr.get(b_col, b_col.replace('label_', '').upper())
                    ax_b.set_xlabel(f"{b_title}: {s_data['branches'][b_col]['prob']:.2f}", fontsize=10.5, fontweight='bold')

                plt.suptitle(f"{tr['fig2_title']} ({sid})", fontsize=13, fontweight='bold', y=0.98)
                plt.tight_layout()
                for ext in ['png', 'pdf']:
                    fig2.savefig(os.path.join(lang_dir, f"Paper_Fig2_MultiBranch_{sid}_{lang}.{ext}"), bbox_inches='tight', facecolor='white')
                plt.close(fig2)

            # 导出每一张独立单图 (含三语标注版 + 无字纯图版)
            single_lang_dir = os.path.join(lang_dir, 'single_panels', sid)
            os.makedirs(single_lang_dir, exist_ok=True)

            def _save_single_panel(img_arr, caption, out_prefix, is_he=False):
                fig_s, ax_s = plt.subplots(1, 1, figsize=(3.6, 3.8), dpi=350, facecolor='white')
                ax_s.imshow(img_arr)
                ax_s.set_xticks([]); ax_s.set_yticks([])
                for sp in ax_s.spines.values(): sp.set_visible(False)
                if is_he and s_data['bar_text'] and s_data['bar_pos']:
                    bx, by = s_data['bar_pos']
                    ax_s.text(bx, by + img_arr.shape[0] * 0.045, s_data['bar_text'],
                              ha='center', va='top', fontsize=9, fontweight='bold', color='#111111')
                ax_s.set_xlabel(caption, fontsize=11, fontweight='bold', labelpad=6)
                for ext in ['png', 'pdf']:
                    fig_s.savefig(os.path.join(single_lang_dir, f"{out_prefix}_{lang}.{ext}"),
                                  bbox_inches='tight', pad_inches=0.04, facecolor='white')
                plt.close(fig_s)

            cls_label = tr.get(s_data['target_cls'], s_data['target_cls'].replace('label_', ''))
            _save_single_panel(s_data['he_img'], f"{tr['he_title']} - {cls_label}", "00_HE_Original", is_he=True)

            # 导出带文字标注的单图
            for m_key in model_order:
                disp_name = tr['ens_name'] if m_key == 'Ensemble' else m_key
                cap_str = f"{disp_name}: {s_data['probs'][m_key]:.2f}"
                clean_m = m_key.replace('.', '_').replace('-', '_')
                _save_single_panel(s_data['clam_maps'][m_key], cap_str, f"02_CLAM_Overlay_Jet_{clean_m}")
                _save_single_panel(s_data['clam_maps_bwr'][m_key], cap_str, f"02_CLAM_Overlay_BlueRed_{clean_m}")

            for b_col, b_info in s_data['branches'].items():
                b_title = tr.get(b_col, b_col.replace('label_', '').upper())
                b_cap = f"{b_title}: {b_info['prob']:.2f}"
                clean_b = b_col.replace('label_', '')
                _save_single_panel(b_info['overlay'], b_cap, f"04_Branch_Overlay_Jet_{clean_b}")
                _save_single_panel(b_info['overlay_bwr'], b_cap, f"04_Branch_Overlay_BlueRed_{clean_b}")

            # 导出无字纯图版
            if lang == args.langs[0]:
                raw_dir = os.path.join(args.output_dir, 'raw_single_images', sid)
                os.makedirs(raw_dir, exist_ok=True)
                plt.imsave(os.path.join(raw_dir, "00_HE_Original.png"), s_data['he_img'])
                for m_key in model_order:
                    clean_m = m_key.replace('.', '_').replace('-', '_')
                    plt.imsave(os.path.join(raw_dir, f"02_CLAM_Overlay_Jet_{clean_m}.png"), s_data['clam_maps'][m_key])
                    plt.imsave(os.path.join(raw_dir, f"02_CLAM_Overlay_BlueRed_{clean_m}.png"), s_data['clam_maps_bwr'][m_key])
                for b_col, b_info in s_data['branches'].items():
                    clean_b = b_col.replace('label_', '')
                    plt.imsave(os.path.join(raw_dir, f"04_Branch_Overlay_Jet_{clean_b}.png"), b_info['overlay'])
                    plt.imsave(os.path.join(raw_dir, f"04_Branch_Overlay_BlueRed_{clean_b}.png"), b_info['overlay_bwr'])

    print(f"\n🎉 方案 B (CLAM 高密重叠采样 + 基准分位数校准) 全套图集生成完毕！保存在: {args.output_dir}/")

if __name__ == "__main__":
    main()