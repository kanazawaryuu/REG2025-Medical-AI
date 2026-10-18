import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import glob
import time
import shutil
import random
import multiprocessing
import json
import argparse
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any

import cv2
import numpy as np
# pyrefly: ignore [import-error]
import pandas as pd
# pyrefly: ignore [import-error]
import torch
# pyrefly: ignore [import-error]
import timm
# pyrefly: ignore [import-error]
import openslide
from PIL import Image, PngImagePlugin, ImageDraw
# pyrefly: ignore [missing-import]
from torchvision import transforms
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# 增加 PIL 图像大小限制，防止处理超大图像时报错
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

# --- 1. 配置管理 ---

@dataclass
class AppConfig:
    """
    应用程序配置类
    集中管理所有路径、参数和阈值，方便统一修改。
    """
    # 路径配置
    wsi_input_dir: Path = Path("/home/zjin/wsi_dataset")  
    features_output_dir: Path = Path("/home/zjin/Extracted_Features_20x") 
    status_cache_path: Path = Path("~/projects/wsi_status_cache.csv").expanduser() 
    json_path: Path = Path(__file__).resolve().parent / "train.json"

    # 支持 H-optimus-1 (1536维), Virchow2 (1280维), CONCH (512维), CONCH 1.5 (768维), GigaPath (1536维)
    model_name: str = "gigapath"  # 可选: "h-optimus-1", "virchow2", "conch", "conch-1.5", "gigapath", "prov-gigapath"

    # 处理参数
    # [关键调整] 数据集本身已固定为 20x，下采样倍率设为 1，严禁降级到 10x
    virtual_downsample_factor: int = 1  
    tile_size: int = 224               # 瓦片大小 224x224
    input_size: int = 224              # 模型输入 224x224 (物理 region_size = 224 * 1 = 224)
    max_tiles_per_wsi: int = 120000    # <--- 改为 120000 (对齐 DGX 版，不截断大图)
    min_tiles_per_wsi: int = 8         # 每个 WSI 最少需要的有效图块数 (少于此数则视为无效样本)
    batch_size: int = 256              # <--- 默认直接设为 256
    num_workers_dataloader: int = 12   # <--- DGX 升级为 12 进程并发读图
    num_workers_tiling: int = 8        # <--- 升级为 8 核
    scan_timeout_seconds: int = 180 # 扫描单张切片的超时时间 (秒)
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # 计算设备

    # 图像处理阈值 (用于过滤背景和低质量图块)
    white_intensity_threshold: int = 230 # 白色背景阈值 (高于此值视为背景)
    saturation_threshold: int = 5 # 饱和度阈值 (低于此值视为背景)
    blur_threshold: int = 50 # 模糊度阈值 (拉普拉斯方差低于此值视为模糊)
    bg_ratio_threshold: float = 0.90 # 图块中背景像素占比阈值 (超过 90% 为背景则丢弃)
    black_threshold: int = 5 # 黑色边缘阈值
    dark_pixel_percentage: float = 0.05 # 黑色像素占比阈值

    def __post_init__(self):
        # 初始化后自动将字符串路径转换为 Path 对象，确保类型安全
        self.wsi_input_dir = Path(self.wsi_input_dir)
        self.features_output_dir = Path(self.features_output_dir)
        self.status_cache_path = Path(self.status_cache_path)
        self.json_path = Path(self.json_path)

# --- 超时控制 (Linux/WSL2 原生 signal.alarm 熔断) ---
import signal

class ScanTimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise ScanTimeoutException("切片扫描超时，强制熔断！")

# --- 2. 日志系统 ---

def setup_logging(output_dir: Path) -> logging.Logger:
    """
    配置日志记录器
    同时输出到控制台和日志文件，方便实时查看和事后排查。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "processing.log"
    
    # 清除之前的 handlers，防止重复日志
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # 创建 Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 1. 文件处理器：详细格式 (带时间戳)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 2. 控制台处理器：简洁格式 (仅消息，无前缀)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 3. 屏蔽第三方库的 INFO 日志 (去除洋文)
    logging.getLogger("timm").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__) # 初始化模块级 logger

# --- 3. 辅助类与函数 ---

def fix_ycbcr_to_pure_white(img_rgb: np.ndarray, white_threshold: int = 228) -> np.ndarray:
    """方案 1 增强版: YCbCr 逆变换 + 自动白平衡拉伸"""
    ycbcr = img_rgb.astype(np.float32)
    Y  = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1] - 128.0
    Cr = ycbcr[:, :, 2] - 128.0

    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    fixed = np.clip(np.stack([R, G, B], axis=-1), 0, 255)

    scale = 255.0 / float(white_threshold)
    fixed_white = np.clip(fixed * scale, 0, 255).astype(np.uint8)
    mask_bg = (fixed_white.mean(axis=-1) >= 240) & (np.ptp(fixed_white, axis=-1) <= 15)
    fixed_white[mask_bg] = 255
    return fixed_white

def check_slide_is_corrupted(slide: openslide.OpenSlide) -> bool:
    """
    鲁棒嗅探：
    1. 多层切片检测 64x64 全景缩略图
    2. 单层切片采样四角边缘玻璃区域 (杜绝组织实质干扰)
    3. 判定阈值严格锁定为 0.20
    """
    try:
        if slide.level_count > 1:
            thumb_mini = np.array(slide.get_thumbnail((64, 64)).convert("RGB"))
        else:
            w, h = slide.dimensions
            # 在切片边缘采样 256x256 纯背景区，杜绝中心肿瘤强伊红假阳性
            pad_x, pad_y = min(100, max(0, w - 300)), min(100, max(0, h - 300))
            patch = slide.read_region((pad_x, pad_y), 0, (256, 256)).convert("RGB")
            thumb_mini = np.array(patch.resize((64, 64)))

        r = thumb_mini[:, :, 0]
        g = thumb_mini[:, :, 1]
        b = thumb_mini[:, :, 2]
        
        # 珊瑚红标准特征：R 高光 (>195)，G/B 显著被锁死在低位 (<160)
        coral_ratio = np.mean((r > 195) & (g < 160) & (b < 160))
        
        # 真正受损片背景占绝大多数 (>50%)，设置 0.20 安全阈值
        return bool(coral_ratio > 0.20)
    except Exception:
        return False

def create_pen_artifact_mask(
    thumb_rgb: np.ndarray, 
    gray: np.ndarray, 
    s_chan: np.ndarray, 
    texture_energy: np.ndarray
) -> np.ndarray:
    """
    基于色彩与模糊度的伪影/污渍截杀函数：
    1. 组织保底颜色 (粉色/紫色)：免除模糊过滤，绝不当成伪影
    2. 伪影一票否决：纯红、纯蓝、纯黑、中性灰污渍、以及非组织颜色的模糊区域
    """
    r = thumb_rgb[:, :, 0].astype(np.float32)
    g = thumb_rgb[:, :, 1].astype(np.float32)
    b = thumb_rgb[:, :, 2].astype(np.float32)
    color_diff = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    # 1. 【组织保底色】：只要落入粉色或紫色，无视模糊度，绝不进入伪影名单
    is_pink_tissue = (r > g + 12) & (s_chan >= 12) & (gray <= 232)
    is_purple_tissue = (b > g + 6) & (s_chan >= 12) & (gray <= 225)
    is_true_tissue = is_pink_tissue | is_purple_tissue

    # 2. 【纯色马克笔与灰色污渍判定】
    pure_red = (r > 120) & (r > 1.4 * b) & (s_chan > 60)          # 纯红笔
    pure_blue = (b > r + 25) & (b > g + 15) & (s_chan > 40)       # 纯蓝笔
    pure_black = (gray < 75) & (s_chan < 35)                       # 纯黑笔
    grey_smudge = (gray < 232) & (s_chan < 20) & (color_diff < 20) # 灰色笔迹、玻片灰尘与中性污点

    # 3. 【模糊判定】：微观无颗粒 (texture < 1.8) 且 不具有真组织颜色
    blur_artifact = (texture_energy < 1.8) & (~is_true_tissue)

    # 汇总所有需要剔除的伪影目标
    raw_artifacts = pure_red | pure_blue | pure_black | grey_smudge | blur_artifact

    # 4. 外扩 17 像素吞掉半透明渐变过渡毛边
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    dilated_artifacts = cv2.dilate(raw_artifacts.astype(np.uint8), kernel, iterations=1) > 0

    return dilated_artifacts

@dataclass
class Tile:
    """简单的数据类，表示一个图块的左上角坐标 (x, y)"""
    x: int
    y: int

    def to_dict(self) -> Dict[str, int]:
        return {'x': self.x, 'y': self.y}

def tile_worker_func(args: Tuple[Path, np.ndarray, int, int, int, AppConfig, bool]) -> List[Dict[str, int]]:
    """
    切片工作进程函数 (独立函数以便 multiprocessing 序列化)
    负责处理 WSI 的一部分区域，筛选出有效的图块。
    """
    # [新增] 禁用 OpenCV 内部多线程，彻底消除进程池 CPU 上下文切换争抢
    if cv2 is not None:
        cv2.setNumThreads(0)
        
    wsi_path, y_coords_chunk, level, region_size, tile_size, cfg, is_corrupted = args
    local_tiles = []
    slide = None
    
    try:
        # 在子进程中打开 Slide
        slide = openslide.OpenSlide(str(wsi_path))
        level_w, level_h = slide.dimensions
        
        # 遍历分配给该进程的 Y 坐标块
        for y in y_coords_chunk:
            for x in range(0, level_w, region_size):
                # 确保瓦片完全落在切片有效图像尺寸内，杜绝越界读出黑边
                if (x + region_size > level_w) or (y + region_size > level_h):
                    continue
                try:
                    # 读取区域并在必要时缩放，避免无意义的 Lanczos 算力空转
                    large_region_pil = slide.read_region((x, y), level, (region_size, region_size)).convert("RGB")
                    if region_size != tile_size:
                        tile_pil = large_region_pil.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
                    else:
                        tile_pil = large_region_pil
                    tile_np = np.array(tile_pil)

                    # 【新增：动态拦截】如果是异常切片，在 QC 过滤前先修复颜色
                    if is_corrupted:
                        tile_np = fix_ycbcr_to_pure_white(tile_np)

                    # --- 质量控制 (QC) 逻辑 ---
                    
                    # 1. 计算灰度 (用于亮度判断)
                    tile_float = tile_np.astype(np.float32)
                    gray_tile = np.dot(tile_float, [0.2989, 0.5870, 0.1140])

                    # 2. 计算饱和度 (用于区分组织和背景)
                    c_max = tile_float.max(axis=2)
                    c_min = tile_float.min(axis=2)
                    delta = c_max - c_min
                    saturation = np.zeros_like(c_max)
                    mask_nonzero = c_max > 0
                    saturation[mask_nonzero] = (delta[mask_nonzero] / c_max[mask_nonzero]) * 255

                    # 3. 背景过滤 (过白且低饱和度)
                    is_true_background = (gray_tile > cfg.white_intensity_threshold) & (saturation < cfg.saturation_threshold)
                    bg_ratio = is_true_background.sum() / (tile_size * tile_size)
                    
                    if bg_ratio > cfg.bg_ratio_threshold:
                        continue # 背景占比过高，丢弃

                    # 4. 黑边过滤 (扫描仪边缘)
                    dark_pixels_ratio = (gray_tile < cfg.black_threshold).sum() / (tile_size * tile_size)
                    if dark_pixels_ratio > cfg.dark_pixel_percentage:
                        continue # 黑边占比过高，丢弃
                    
                    # 5. 模糊过滤 (可选，依赖 cv2)
                    if cv2 is not None:
                        gray_cv = cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
                        blur_score = cv2.Laplacian(gray_cv, cv2.CV_64F).var()
                        if blur_score < cfg.blur_threshold:
                            continue # 图像过糊，丢弃

                    # 6. 【新增】单个瓦片红蓝油墨兜底检测
                    r_t = tile_float[:, :, 0]
                    g_t = tile_float[:, :, 1]
                    b_t = tile_float[:, :, 2]
                    # 若瓦片内包含 >15% 的高纯度蓝笔或红笔墨水，直接丢弃
                    is_ink_px = ((b_t > r_t + 35) & (b_t > g_t + 20) & (saturation > 75)) | \
                                ((r_t > 130) & (r_t > 1.8 * b_t) & (b_t < 95) & (saturation > 135))
                    if (is_ink_px.sum() / (tile_size * tile_size)) > 0.15:
                        continue

                    # 通过所有检查，保留该图块坐标（确保落在有效物理边缘内）
                    if (x + region_size <= level_w) and (y + region_size <= level_h):
                        local_tiles.append({'x': x, 'y': y})

                except Exception:
                    continue # 跳过单个图块的读取错误
                    
    except Exception as e:
        # 在子进程中打印错误，因为 logger 可能未配置多进程安全
        print(f"Worker Error: {e}")
    finally:
        if slide: slide.close() # 确保关闭文件句柄
        
    return local_tiles

class WSITileDataset(Dataset):
    """
    PyTorch 数据集类
    用于 DataLoader 并行读取和预处理 WSI 图块，提升 GPU 利用率。
    """
    def __init__(self, wsi_path: Path, tiles: List[Dict[str, int]], region_size: int, tile_size: int, 
                 transform=None, is_corrupted: bool = False):
        self.wsi_path = str(wsi_path)
        self.tiles = tiles
        self.region_size = region_size
        self.tile_size = tile_size
        self.transform = transform
        self.is_corrupted = is_corrupted  # 【新增】
        self._slide = None # 线程局部 slide 句柄，延迟初始化

    def _get_slide(self):
        """延迟加载 OpenSlide 对象，每个 worker 线程一个实例"""
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.wsi_path)
        return self._slide

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        t = self.tiles[idx]
        try:
            slide = self._get_slide()
            img = slide.read_region((t['x'], t['y']), 0, (self.region_size, self.region_size)).convert("RGB")
            
            # 【新增：动态拦截】仅在异常片激活，正常片直接跳过 (耗时 0)
            if self.is_corrupted:
                img_np = fix_ycbcr_to_pure_white(np.array(img))
                img = Image.fromarray(img_np)

            # 只有当原始尺寸和模型输入不一致时才缩放，避免无意义的 Lanczos 算力空转
            if self.region_size != self.tile_size:
                img = img.resize((self.tile_size, self.tile_size), Image.Resampling.BILINEAR)
        except Exception as e:
            logger.error(f"Error reading tile {t}: {e}")
            img = Image.new("RGB", (self.tile_size, self.tile_size), (255, 255, 255))
            
        if self.transform:
            img = self.transform(img)
        return img 

    def __del__(self):
        try:
            if hasattr(self, '_slide') and self._slide is not None:
                self._slide.close()
        except Exception:
            pass

def worker_init_fn(worker_id):
    import warnings
    warnings.filterwarnings("ignore")

# --- 4. 核心管理器类 ---

class WSIManager:
    """
    资源管家类
    负责文件扫描、筛选和状态管理 (CSV 缓存)。
    """
    def __init__(self, config: AppConfig):
        self.config = config
        self.wsi_to_organ_map = self._load_json_map()
        self.status_cache = self._load_status_cache()

    def _load_json_map(self) -> Dict[str, str]:
        """加载 JSON 元数据，建立 ID 到器官的映射"""
        mapping = {}
        if not self.config.json_path.exists():
            logger.warning(f"JSON file not found: {self.config.json_path}")
            return mapping
            
        try:
            with open(self.config.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                if 'id' in item and 'report' in item and item['report']:
                    organ = item['report'].split(',')[0].strip()
                    raw_id = str(item['id']).strip()
                    mapping[raw_id] = organ
                    # 同时匹配 stem，防止无后缀 ID 导致归入 Unknown
                    mapping[Path(raw_id).stem] = organ
            logger.info(f"已加载 {len(mapping)} 个样本映射。")
        except Exception as e:
            logger.error(f"JSON读取错误: {e}")
        return mapping

    def _load_status_cache(self) -> Dict[str, str]:
        # 每个模型独享独立的缓存文件
        cache_file = self.config.status_cache_path.parent / f"wsi_status_cache_{self.config.model_name}.csv"
        if cache_file.exists():
            try:
                return pd.read_csv(cache_file).set_index('wsi_filename')['status'].to_dict()
            except Exception:
                return {}
        return {}

    def update_status(self, wsi_filename: str, status: str):
        cache_file = self.config.status_cache_path.parent / f"wsi_status_cache_{self.config.model_name}.csv"
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            if not cache_file.exists():
                df = pd.DataFrame(columns=['wsi_filename', 'status'])
            else:
                df = pd.read_csv(cache_file)
            
            if wsi_filename in df['wsi_filename'].values:
                df.loc[df['wsi_filename'] == wsi_filename, 'status'] = status
            else:
                new_row = pd.DataFrame([{'wsi_filename': wsi_filename, 'status': status}])
                df = pd.concat([df, new_row], ignore_index=True)
            
            df.to_csv(cache_file, index=False)
            self.status_cache[wsi_filename] = status
        except Exception as e:
            logger.error(f"Failed to update status cache: {e}")

    def get_pending_files(self, organ_filter: Optional[Set[str]], id_filter: Optional[Set[str]]) -> List[Tuple[Path, Path]]:
        """
        获取待处理的文件列表 (按模型名分流输出目录)
        """
        all_files = sorted(list(self.config.wsi_input_dir.glob("*.tif*")))
        pending = []

        for path in all_files:
            filename = path.name
            basename = path.stem

            if id_filter and basename not in id_filter:
                continue
            
            organ = self.wsi_to_organ_map.get(basename) or self.wsi_to_organ_map.get(filename, "Unknown")
            if organ_filter and organ not in organ_filter:
                continue

            organ_dir = self.config.features_output_dir / self.config.model_name / organ
            status_key = f"{filename}_{self.config.model_name}"
            
            cached_status = self.status_cache.get(filename) or self.status_cache.get(status_key)
            feature_file = organ_dir / f'{basename}_features_downsampled{self.config.virtual_downsample_factor}x.npy'
            
            # 【核心修复】只要磁盘特征文件存在且非空，直接视为已完成，跳过提取；确认为空样本亦跳过
            if feature_file.exists() and feature_file.stat().st_size > 0:
                continue
            if cached_status == 'no_tiles':
                continue

            pending.append((path, organ_dir))
        
        return pending

class WSIProcessor:
    """
    核心处理器类
    封装了切片、特征提取和结果保存的完整流水线。
    """
    def __init__(self, config: AppConfig):
        self.config = config
        self.model, self.transform = self._load_model_and_transform()
        
        # 准备输出目录
        self.config.features_output_dir.mkdir(parents=True, exist_ok=True)

    def _load_model_and_transform(self):
        """动态加载 H-optimus-1, Virchow2 或 CONCH 及其配套预处理"""
        logger.info(f"正在加载基础模型: {self.config.model_name} ...")
        
        if self.config.model_name in ["h-optimus-1", "h_optimus_1", "hoptimus1"]:
            # 1. H-optimus-1 骨干网络加载 (timm, Hugging Face: bioptimus/H-optimus-1)
            # 输出特征维度固定为 1536
            model = timm.create_model(
                "hf-hub:bioptimus/H-optimus-1", 
                pretrained=True, 
                init_values=1e-5, 
                dynamic_img_size=False
            )
            model.to(self.config.device).eval()
            
            # H-optimus-1 官方标准化参数 (RGB: mean=[0.707223, 0.578729, 0.703617], std=[0.211883, 0.230117, 0.177517])
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617), 
                    std=(0.211883, 0.230117, 0.177517)
                )
            ])
            return model, transform

        elif self.config.model_name == "virchow2":
            # 2. Virchow2 加载 (timm)
            model = timm.create_model(
                "hf_hub:paige-ai/Virchow2", 
                pretrained=True, 
                mlp_layer=timm.layers.SwiGLUPacked, 
                act_layer=torch.nn.SiLU
            )
            model.to(self.config.device).eval()
            
            # 标准 ImageNet 归一化 (直接一步到位，坚决摒弃 CenterCrop 与二次缩放)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            return model, transform

        elif self.config.model_name == "conch":
            # 3. CONCH 加载 (使用官方库)
            from conch.open_clip_custom import create_model_from_pretrained
            model, conch_transform = create_model_from_pretrained(
                model_cfg="conch_ViT-B-16", 
                checkpoint_path="hf_hub:MahmoodLab/CONCH"
            )
            model.to(self.config.device).eval()
            return model, conch_transform

        elif self.config.model_name in ["conch-1.5", "conch1.5"]:
            # 4. CONCH 1.5 加载 (ViT-L/16, 输入 448x448, 官方纯视觉权重)
            # 优先从本地 Hugging Face 缓存直读，彻底免去重新下载与库版本冲突
            ckpt_path = Path("~/.cache/huggingface/hub/models--MahmoodLab--conchv1_5").expanduser()
            bin_files = list(ckpt_path.glob("**/pytorch_model_vision.bin"))
            
            if bin_files:
                weights_path = str(bin_files[0])
                logger.info(f"⚡ [本地命中] 直接加载已下载的 CONCH-v1.5 权重: {weights_path}")
                # 依据 pos_embed (1, 785, 1024) 实例化 448 分辨率的 ViT-Large
                model = timm.create_model(
                    "vit_large_patch16_224", 
                    img_size=448, 
                    patch_size=16, 
                    init_values=1.0, 
                    num_classes=0, 
                    dynamic_img_size=True
                )
                # 读取并剥离 'trunk.' 前缀后载入
                state_dict = torch.load(weights_path, map_location="cpu")
                cleaned_sd = {k.replace("trunk.", ""): v for k, v in state_dict.items() if k.startswith("trunk.")}
                model.load_state_dict(cleaned_sd, strict=False)
            else:
                # 备用容错：直接走 timm 原生 hub 加载
                model = timm.create_model("hf-hub:MahmoodLab/conchv1_5", pretrained=True)

            model.to(self.config.device).eval()

            # 官方标准预处理 (ImageNet 均值/方差 + 双三次插值缩放到 448x448)
            transform = transforms.Compose([
                transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            return model, transform

        elif self.config.model_name in ["gigapath", "prov-gigapath"]:
            # 5. Prov-GigaPath 骨干网络加载 (timm, 输出 1536 维特征)
            model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            model.to(self.config.device).eval()

            # 官方标准 ImageNet 归一化 (直接接收 224x224 瓦片)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            return model, transform

        else:
            raise ValueError(f"不支持的模型: {self.config.model_name}，支持 'h-optimus-1', 'virchow2', 'conch', 'conch-1.5', 'gigapath'")

    def process_wsi(self, wsi_path: Path, output_dir: Path) -> Tuple[bool, str]:
        """
        处理单个 WSI 文件的完整流程 (纯串行，磁盘独占)
        """
        basename = wsi_path.stem
        start_time = time.time()
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # 获取切片基本信息 + 一次性探测色彩异常
            with openslide.OpenSlide(str(wsi_path)) as slide:
                w, h = slide.dimensions
                orig_mpp = slide.properties.get('openslide.mpp-x', 'unknown')
                is_corrupted = check_slide_is_corrupted(slide)

            if is_corrupted:
                logger.warning(f"⚠️ [色彩自愈] 检测到切片 [{wsi_path.name}] 存在 YCbCr 色彩解码异常，已激活动态逆变换白平衡修复！")

            region_size = self.config.tile_size * self.config.virtual_downsample_factor
            
            # 1. 优先精准探测是否已有模型提取过坐标 (O(1) 路径探测，杜绝 glob("**/*") 递归导致的严重 I/O 延迟)
            coord_filename = f"{basename}_coords_downsampled{self.config.virtual_downsample_factor}x.npy"
            organ_name = output_dir.name
            coord_candidates = [
                (output_dir / coord_filename, self.config.model_name),
                (self.config.features_output_dir / "virchow2" / organ_name / coord_filename, "virchow2"),
                (self.config.features_output_dir / "conch_1.5" / organ_name / coord_filename, "conch_1.5"),
                (self.config.features_output_dir / "conch-1.5" / organ_name / coord_filename, "conch-1.5"),
                (self.config.features_output_dir / "h_optimus_1_2025" / "h-optimus-1" / organ_name / coord_filename, "h-optimus-1"),
                (self.config.features_output_dir / "h-optimus-1" / organ_name / coord_filename, "h-optimus-1"),
                (self.config.features_output_dir / "conch" / organ_name / coord_filename, "conch"),
                (self.config.features_output_dir / "gigapath" / organ_name / coord_filename, "gigapath"),
                (self.config.features_output_dir / organ_name / coord_filename, "legacy"),
            ]
            
            found_coord = None
            source_model = ""
            for p, m in coord_candidates:
                if p.is_file():
                    found_coord = p
                    source_model = m
                    break

            if found_coord:
                coords_loaded = np.load(found_coord)
                final_tiles = [{'x': int(pt[0]), 'y': int(pt[1])} for pt in coords_loaded]
                scan_time = 0.0
                logger.info(f"⚡ [坐标复用] 检测到已有切片坐标 ({len(final_tiles)} 个瓦片，源自 {source_model})，空间物理严格对齐，跳过 CPU 扫描！")
            else:
                # 首次提取：使用 signal.alarm 施加硬超时防护
                scan_start = time.time()
                has_alarm = hasattr(signal, 'SIGALRM') and hasattr(signal, 'alarm')
                if has_alarm:
                    signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(self.config.scan_timeout_seconds)  # 设定 180 秒闹钟
                
                try:
                    valid_tiles = self._scan_tiles(wsi_path, h, region_size, is_corrupted=is_corrupted)
                    final_tiles = self._sample_tiles(valid_tiles)
                except ScanTimeoutException:
                    logger.error(f"⏰ [扫描超时] {basename} 扫描耗时超过 {self.config.scan_timeout_seconds}s，强制跳过！")
                    return False, 'scan_timeout'
                finally:
                    if has_alarm:
                        signal.alarm(0)  # 无论成功或失败，立刻解除闹钟
                    
                scan_time = time.time() - scan_start

            # 核心收敛：无论坐标是复用的还是新采样的，统一检查最小瓦片门槛
            if len(final_tiles) < self.config.min_tiles_per_wsi:
                logger.warning(f"⏩ [跳过] {basename}: 有效图块不足 ({len(final_tiles)} < {self.config.min_tiles_per_wsi})")
                return False, 'no_tiles'
            
            # 2. DataLoader 全力把图喂给 GPU (此时磁盘独占，读取速度拉满)
            extract_start = time.time()
            final_feats = self._extract_features(wsi_path, final_tiles, region_size, is_corrupted=is_corrupted)
            extract_time = time.time() - extract_start
            
            if final_feats is None:
                logger.error(f"❌ [提取失败] {basename}: 模型返回空特征")
                return False, 'extraction_failed'

            # 【核心修复】数值安全熔断，杜绝半精度下产生 NaN 污染下游训练
            if np.isnan(final_feats).any():
                logger.error(f"❌ [数值异常] {basename}: 提取特征检测到 NaN，拒绝入库！")
                return False, 'nan_detected'

            # 3. 保存特征与状态 (原子操作，单线程无锁写入)
            self._save_results(basename, final_feats, final_tiles, w, h, region_size, orig_mpp, wsi_path, output_dir, is_corrupted=is_corrupted)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ 完成: {basename} | 形状: {final_feats.shape} | 扫描: {scan_time:.1f}s | 提取: {extract_time:.1f}s | 总计: {elapsed:.1f}s")
            return True, 'completed'

        except Exception as e:
            logger.error(f"❌ [处理错误] {basename}: {e}")
            logger.debug(traceback.format_exc())
            return False, 'error' 

    def _scan_tiles(self, wsi_path: Path, height: int, region_size: int, is_corrupted: bool = False) -> List[Dict[str, int]]:
        """纯像素物理判据切分：坚决不用形态学闭合，形状任意扭曲均忠实保留"""
        with openslide.OpenSlide(str(wsi_path)) as slide:
            w, h = slide.dimensions
            
            if slide.level_count > 1:
                target_dim = 2048
                scale = max(w, h) / target_dim
                thumb_w_req, thumb_h_req = int(w / scale), int(h / scale)
                thumb = slide.get_thumbnail((thumb_w_req, thumb_h_req)).convert("RGB")
                thumb_np = np.array(thumb)

                # 【核心修复 1】立即用 OpenSlide 返回的真实图像形状覆盖，杜绝 1 像素偏差
                thumb_h, thumb_w = thumb_np.shape[:2]

                if is_corrupted:
                    thumb_np = fix_ycbcr_to_pure_white(thumb_np)

                hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
                s_channel = hsv[:, :, 1]
                gray_np = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2GRAY)

                # 1. 微观颗粒度计算
                gray_f = gray_np.astype(np.float32)
                blur_micro = cv2.GaussianBlur(gray_f, (3, 3), 0)
                granularity = np.abs(gray_f - blur_micro)
                texture_energy = cv2.blur(granularity, (7, 7))

                r_chan = thumb_np[:, :, 0].astype(np.float32)
                g_chan = thumb_np[:, :, 1].astype(np.float32)
                b_chan = thumb_np[:, :, 2].astype(np.float32)
                c_max = np.maximum(np.maximum(r_chan, g_chan), b_chan)
                c_min = np.minimum(np.minimum(r_chan, g_chan), b_chan)
                color_diff = c_max - c_min

                # 2. 【组织保底准入】恢复纯净准入，绝不在真组织里误伤低纹理平滑基质
                is_pink_tissue = (r_chan > g_chan + 3) & (r_chan >= b_chan - 4) & (s_channel >= 6) & (gray_np <= 236)
                is_purple_tissue = (b_chan > g_chan + 3) & (r_chan > g_chan - 2) & (s_channel >= 6) & (gray_np <= 228)
                real_tissue = is_pink_tissue | is_purple_tissue

                # 3. 【记号笔与污渍截杀】
                # (a) 红笔
                pure_red = (r_chan > 130) & (r_chan > 1.60 * b_chan) & (r_chan > g_chan + 30) & (b_chan < 110) & (s_channel > 45)
                # (b) 蓝笔
                pure_blue = (b_chan > r_chan + 25) & (b_chan > g_chan + 15) & (s_channel > 40) & (
                    (r_chan <= g_chan + 8) | (texture_energy < 3.0)
                )
                # (c) 黑笔
                pure_black = (gray_np < 75) & (color_diff < 25) & (texture_energy < 3.0)

                # (d) 灰笔截杀：
                # 【核心修复 1】只要属于 real_tissue，绝不允许被判定为 grey_on_glass，彻底根治浅粉组织内部误伤！
                grey_on_glass = (gray_np < 228) & (color_diff < 22) & (texture_energy < 2.2) & (r_chan <= g_chan + 10) & (~real_tissue)
                grey_on_tissue = (c_max < 178) & (color_diff < 46) & (texture_energy < 2.95)
                grey_marker = grey_on_glass | grey_on_tissue

                pen_strokes = pure_red | pure_blue | pure_black | grey_marker

                # 笔迹隔离外扩（保持 27 像素安全净空）
                PEN_MARGIN = 27
                open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                clean_pen = cv2.morphologyEx(pen_strokes.astype(np.uint8), cv2.MORPH_OPEN, open_kernel)
                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PEN_MARGIN, PEN_MARGIN))
                dilated_pen = cv2.dilate(clean_pen, dilate_kernel, iterations=1) > 0

                # 4. 生成组织掩码
                raw_mask = np.zeros_like(gray_np, dtype=np.uint8)
                raw_mask[real_tissue & (~dilated_pen)] = 255
                raw_mask[gray_np < 20] = 0

                # 【核心修复 2】滤除漂浮微小碎屑 (解决图 11)：连通域面积小于 350 像素的碎屑直接抹除
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask)
                mask = np.zeros_like(raw_mask)
                for label_idx in range(1, num_labels):
                    if stats[label_idx, cv2.CC_STAT_AREA] >= 350:  # 过滤面积过小的孤立浮屑
                        mask[labels == label_idx] = 255

                # 5. 瓦片收集 (维持对笔迹禁区的零容忍安全机制)
                thumb_step = region_size / scale
                valid_tiles = []
                for y in range(0, h, region_size):
                    ty = int(y / scale)
                    ty_end = min(int(ty + thumb_step), thumb_h)
                    for x in range(0, w, region_size):
                        tx = int(x / scale)
                        tx_end = min(int(tx + thumb_step), thumb_w)
                        sub_mask = mask[ty:ty_end, tx:tx_end]
                        sub_pen = dilated_pen[ty:ty_end, tx:tx_end]
                        if sub_mask.size > 0:
                            # 碰上任何笔迹隔离区一票否决
                            if np.count_nonzero(sub_pen) > 0:
                                continue
                            if (np.count_nonzero(sub_mask) / sub_mask.size) >= 0.08:
                                if (x + region_size <= w) and (y + region_size <= h):
                                    valid_tiles.append({'x': x, 'y': y})
                return valid_tiles

        # --- 情况 B: 扁平单层 TIFF (level_count == 1)，避免 get_thumbnail 爆内存 ---
        # 回退至安全的多进程瓦片读取 (使用已有的 tile_worker_func)
        y_coords = [y for y in range(0, height, region_size) if y + region_size <= height]
        y_chunks = np.array_split(np.array(y_coords), self.config.num_workers_tiling)
        worker_args = [
            (wsi_path, chunk, 0, region_size, self.config.tile_size, self.config, is_corrupted) 
            for chunk in y_chunks if len(chunk) > 0
        ]
        
        valid_tiles = []
        with multiprocessing.Pool(processes=self.config.num_workers_tiling) as pool:
            for result in pool.imap_unordered(tile_worker_func, worker_args):
                valid_tiles.extend(result)
        # 确保瓦片完全落在切片有效图像尺寸内
        valid_tiles = [t for t in valid_tiles if (t['x'] + region_size <= w) and (t['y'] + region_size <= h)]
        return valid_tiles

    def _sample_tiles(self, tiles: List[Dict[str, int]], seed: int = 42) -> List[Dict[str, int]]:
        """确定性采样：确保不同模型对同一张切片抽取的瓦片绝对一致，消除空间不对齐"""
        # 1. 先进行确定性排序（消除 imap_unordered 多进程返回结果的无序性）
        tiles.sort(key=lambda t: (t['y'], t['x']))
        
        if len(tiles) <= self.config.max_tiles_per_wsi:
            logger.info(f"保留所有 {len(tiles)} 个图块。")
            return tiles
        else:
            # 2. 锁定随机种子进行抽样
            logger.info(f"从 {len(tiles)} 个图块中确定性抽取 {self.config.max_tiles_per_wsi} 个 (seed={seed})。")
            rng = random.Random(seed)
            final = rng.sample(tiles, self.config.max_tiles_per_wsi)
            final.sort(key=lambda t: (t['y'], t['x']))
            return final

    def _extract_features(self, wsi_path: Path, tiles: List[Dict[str, int]], region_size: int, is_corrupted: bool = False) -> Optional[np.ndarray]:
        """特征提取：严格遵循论文规范，Virchow2 仅保留纯 Class Token"""
        dataset = WSITileDataset(wsi_path, tiles, region_size, self.config.tile_size, 
                                 transform=self.transform, is_corrupted=is_corrupted)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_dataloader,
            prefetch_factor=3,        # 让 CPU 提前在后台解压好下 3 个批次的图片
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,  # 保持 worker 进程常驻，消除反复创建进程的开销
            worker_init_fn=worker_init_fn
        )

        all_feats = []
        device_type = "cuda" if torch.cuda.is_available() and "cuda" in str(self.config.device) else "cpu"
        use_amp = (device_type == "cuda")

        # 动态检测硬件：优先启用 bfloat16 (8位指数位完全对齐 float32，彻底免疫 ViT 注意力溢出 NaN 陷阱)，否则回退 float16
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        with torch.no_grad():
            for batch_imgs in tqdm(loader, desc=f"提取特征 ({self.config.model_name})", leave=False):
                batch_imgs = batch_imgs.to(self.config.device)
                
                # 使用自动混合精度加速并大幅节省显存 (防 24G 显存 OOM 与 NaN 溢出)
                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    if self.config.model_name in ["h-optimus-1", "h_optimus_1", "hoptimus1"]:
                        # H-optimus-1: 直接输出 1536 维特征向量
                        output = self.model(batch_imgs)
                        feats = output[0] if isinstance(output, (tuple, list)) else output
                    elif self.config.model_name == "virchow2":
                        # Virchow2: 取 index 0 的 Class Token (1280 维)
                        output = self.model(batch_imgs)
                        feats = output[:, 0]
                    elif self.config.model_name in ["conch-1.5", "conch1.5"]:
                        # CONCH 1.5: 纯视觉 ViT 前向提取 (直接输出特征向量)
                        output = self.model(batch_imgs)
                        feats = output[0] if isinstance(output, (tuple, list)) else output
                    elif self.config.model_name in ["gigapath", "prov-gigapath"]:
                        # GigaPath: 直接输出 1536 维特征向量
                        output = self.model(batch_imgs)
                        feats = output[0] if isinstance(output, (tuple, list)) else output
                    elif self.config.model_name == "conch":
                        # CONCH v1.0: 跨模态潜空间图块特征
                        feats = self.model.encode_image(batch_imgs, proj_contrast=True, normalize=False)
                    else:
                        raise ValueError(f"不支持的模型: {self.config.model_name}")
                    
                all_feats.append(feats.float().cpu().numpy()) # 转回 float32 保存

        if not all_feats:
            return None
        return np.vstack(all_feats)

    def _save_results(self, basename: str, feats: np.ndarray, tiles: List[Dict[str, int]], 
                     w: int, h: int, region_size: int, mpp: str, wsi_path: Path, output_dir: Path, is_corrupted: bool = False):
        """保存标准特征文件、坐标文件 (给下游 CLAM 热力图直读) 与元数据"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存特征
        np.save(output_dir / f'{basename}_features_downsampled{self.config.virtual_downsample_factor}x.npy', feats)
        
        # 2. 保存坐标 (保持 (N, 2) 的 [x, y] 矩阵，下游 CLAM 热力图直读)
        coords = np.array([[t['x'], t['y']] for t in tiles], dtype=np.int32)
        np.save(output_dir / f'{basename}_coords_downsampled{self.config.virtual_downsample_factor}x.npy', coords)

        # 3. 保存元数据 (记录模型来源与下采样尺寸)
        metadata = {
            'wsi_id': basename,
            'model_name': self.config.model_name,
            'feature_dim': feats.shape[1],
            'original_width': w,
            'original_height': h,
            'num_tiles': len(tiles),
            'patch_size': region_size,
            'mpp': str(mpp),
            'is_color_corrected': bool(is_corrupted)  # 新增审计标记
        }
        with open(output_dir / f'{basename}_meta.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # 4. 生成 QC 图 -> 若已有同名 QC 图则直接复用，避免白白浪费大量 CPU 画图时间
        qc_dir = output_dir / "qc_overlay"
        qc_dir.mkdir(parents=True, exist_ok=True)
        qc_file = qc_dir / f"{basename}_QC.jpg"
        if not qc_file.exists():
            # 精准 O(1) 探测其他模型或历史目录中的 QC 图，杜绝 glob("**/*") 递归扫描
            organ_name = output_dir.name
            qc_candidates = [
                self.config.features_output_dir / "virchow2" / organ_name / "qc_overlay" / f"{basename}_QC.jpg",
                self.config.features_output_dir / "conch_1.5" / organ_name / "qc_overlay" / f"{basename}_QC.jpg",
                self.config.features_output_dir / "conch-1.5" / organ_name / "qc_overlay" / f"{basename}_QC.jpg",
                self.config.features_output_dir / "h_optimus_1_2025" / "h-optimus-1" / organ_name / "qc_overlay" / f"{basename}_QC.jpg",
                self.config.features_output_dir / "h-optimus-1" / organ_name / "qc_overlay" / f"{basename}_QC.jpg",
                self.config.features_output_dir / "conch" / organ_name / "qc_overlay" / f"{basename}_QC.jpg",
                self.config.features_output_dir / "gigapath" / organ_name / "qc_overlay" / f"{basename}_QC.jpg",
                self.config.features_output_dir / organ_name / "qc_overlay" / f"{basename}_QC.jpg",
            ]
            found_qc = next((p for p in qc_candidates if p.is_file()), None)
            if found_qc:
                shutil.copyfile(found_qc, qc_file)
            else:
                self._save_visualization(wsi_path, tiles, qc_dir, basename, region_size, is_corrupted=is_corrupted)

    def _save_visualization(self, wsi_path: Path, tiles: List[Dict[str, int]], output_dir: Path, basename: str, region_size: int, is_corrupted: bool = False):
        """生成并保存可视化质控图：在缩略图上绘制绿色方框 (带单层 TIFF 防爆保护)"""
        try:
            with openslide.OpenSlide(str(wsi_path)) as slide:
                # 若为单层扁平 TIFF，严禁调用 get_thumbnail，直接跳过以保全进程
                if slide.level_count == 1:
                    logger.info(f"切片 {basename} 为扁平单层 TIFF，跳过生成全景 QC 图以防内存溢出。")
                    return

                w, h = slide.dimensions
                target_dim = 2048
                downsample = max(w, h) / target_dim
                thumb_size = (int(w / downsample), int(h / downsample))

                thumbnail = slide.get_thumbnail(thumb_size).convert("RGB")
                # 【直接使用传入的状态】
                if is_corrupted:
                    thumbnail = Image.fromarray(fix_ycbcr_to_pure_white(np.array(thumbnail)))
                draw = ImageDraw.Draw(thumbnail)
                
                scale_x = thumb_size[0] / w
                scale_y = thumb_size[1] / h

                for t in tiles:
                    x, y = t['x'], t['y']
                    x_thumb = int(x * scale_x)
                    y_thumb = int(y * scale_y)
                    w_thumb = int(region_size * scale_x)
                    h_thumb = int(region_size * scale_y)
                    draw.rectangle([x_thumb, y_thumb, x_thumb + w_thumb, y_thumb + h_thumb], outline="green", width=2)

                thumbnail.save(output_dir / f"{basename}_QC.jpg", "JPEG", quality=80)
        except Exception as e:
            logger.warning(f"可视化失败 {basename}: {e}")

# --- 5. 主程序 (纯串行版) ---

def main():
    parser = argparse.ArgumentParser(description="串行极速版: H-optimus-1 / Virchow2 / CONCH / CONCH 1.5 / GigaPath 特征提取 (磁盘独占无冲突)")
    parser.add_argument('--model', type=str, default='gigapath', 
                        choices=['h-optimus-1', 'h_optimus_1', 'virchow2', 'conch', 'conch1.5', 'conch-1.5', 'conch_1_5', 'gigapath', 'prov-gigapath'], 
                        help='选择提取特征的基础模型')
    parser.add_argument('--wsi-dir', type=str, default=None, help='WSI 切片文件夹路径')
    parser.add_argument('--output-dir', type=str, default=None, help='特征保存目录路径')
    parser.add_argument('--batch-size', type=int, default=256, help='特征提取批次大小 (默认 256)')
    parser.add_argument('--organs', type=str, help='按器官筛选')
    parser.add_argument('--ids', type=str, help='按 ID 筛选')
    args = parser.parse_args()

    # 1. 配置与日志
    config = AppConfig()
    raw_model = args.model.lower().replace('_', '-')
    if raw_model in ['conch1.5', 'conch-1.5', 'conch-1-5']:
        config.model_name = 'conch-1.5'
    elif raw_model in ['gigapath', 'prov-gigapath']:
        config.model_name = 'gigapath'
    else:
        config.model_name = raw_model

    # 动态覆盖 DGX 上的真实切片与输出路径
    if args.wsi_dir:
        config.wsi_input_dir = Path(args.wsi_dir)
    if args.output_dir:
        config.features_output_dir = Path(args.output_dir)
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    global logger
    logger = setup_logging(config.features_output_dir / config.model_name)
    logger.info(f"--- 串行极速版 (单卡独占磁盘 I/O) 启动 [模型: {config.model_name}] ---")
    
    try:
        # 2. 初始化资源与处理器
        manager = WSIManager(config)
        processor = WSIProcessor(config) # 加载模型至 GPU

        # 3. 获取待处理任务列表
        organ_filter = {o.strip() for o in args.organs.split(',')} if args.organs else None
        id_filter = {i.strip() for i in args.ids.split(',')} if args.ids else None
        
        files_to_process = manager.get_pending_files(organ_filter, id_filter)
        logger.info(f"📋 待处理任务: {len(files_to_process)} 个文件")
        
        if not files_to_process:
            logger.info("所有文件已处理完毕或无匹配任务，退出。")
            return

        # 4. 纯串行处理流水线 (直接委托给 WSIProcessor.process_wsi)
        start_time_total = time.time()
        for idx, (wsi_path, organ_dir) in enumerate(files_to_process, 1):
            logger.info(f"\n[{idx}/{len(files_to_process)}] 🚀 开始处理: {wsi_path.name}")
            
            try:
                # 直接委托给 WSIProcessor 处理，消除重复代码
                success, status = processor.process_wsi(wsi_path, organ_dir)
                manager.update_status(wsi_path.name, status)
            except Exception as e:
                logger.error(f"❌ [主循环调度错误] {wsi_path.name}: {e}")
                logger.debug(traceback.format_exc())
                manager.update_status(wsi_path.name, 'error')

        total_time = time.time() - start_time_total
        logger.info(f"\n🏁 全部任务完成！总耗时: {total_time:.1f}s (共处理 {len(files_to_process)} 个切片)")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ 用户中断 (Ctrl+C)，正在退出...")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"系统严重错误: {e}")
        logger.debug(traceback.format_exc())

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
