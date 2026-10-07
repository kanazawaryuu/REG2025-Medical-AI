# run_stage1_v4_pipeline.py (V4 纯串行极速版: 单切片磁盘独占 + 多模型解耦)
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
    wsi_input_dir: Path = Path("/mnt/e/REG_train_CLEANED")  
    # 明确标示为 20x 标准数据，隔绝历史旧特征
    features_output_dir: Path = Path("/mnt/e/Extracted_Features_train_20x") 
    status_cache_path: Path = Path("~/projects/wsi_status_cache.csv").expanduser() 
    # 自动定位到脚本同级目录下的 train.json，杜绝相对路径丢失
    json_path: Path = Path(__file__).resolve().parent / "train.json"

    # [更新] 仅保留 virchow2 与 conch，彻底弃用 gigapath
    model_name: str = "conch"  # 可选: "virchow2", "conch"

    # 处理参数
    # [关键调整] 数据集本身已固定为 20x，下采样倍率设为 1，严禁降级到 10x
    virtual_downsample_factor: int = 1  
    tile_size: int = 224               # 瓦片大小 224x224
    input_size: int = 224              # 模型输入 224x224 (物理 region_size = 224 * 1 = 224)
    max_tiles_per_wsi: int = 12000     # 每个 WSI 保留的最大图块数 (超过则随机采样)
    min_tiles_per_wsi: int = 8         # 每个 WSI 最少需要的有效图块数 (少于此数则视为无效样本)
    batch_size: int = 64               # 特征提取时的批次大小 (根据显存调整)
    num_workers_dataloader: int = 6    # WSL2 挂载盘防锁死
    num_workers_tiling: int = 4        # 给切片扫描分配 4 个核
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

@dataclass
class Tile:
    """简单的数据类，表示一个图块的左上角坐标 (x, y)"""
    x: int
    y: int

    def to_dict(self) -> Dict[str, int]:
        return {'x': self.x, 'y': self.y}

def tile_worker_func(args: Tuple[Path, np.ndarray, int, int, int, AppConfig]) -> List[Dict[str, int]]:
    """
    切片工作进程函数 (独立函数以便 multiprocessing 序列化)
    负责处理 WSI 的一部分区域，筛选出有效的图块。
    """
    # [新增] 禁用 OpenCV 内部多线程，彻底消除进程池 CPU 上下文切换争抢
    if cv2 is not None:
        cv2.setNumThreads(0)
        
    wsi_path, y_coords_chunk, level, region_size, tile_size, cfg = args
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
    def __init__(self, wsi_path: Path, tiles: List[Dict[str, int]], region_size: int, tile_size: int, transform=None):
        self.wsi_path = str(wsi_path)
        self.tiles = tiles
        self.region_size = region_size
        self.tile_size = tile_size
        self.transform = transform
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
            
            # 【核心修复】遇到终态（已完成 或 确认为空样本）果断跳过，严禁死循环重复处理
            if cached_status == 'completed' and feature_file.exists():
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
        """动态加载 Virchow2 或 CONCH 及其配套预处理"""
        logger.info(f"正在加载基础模型: {self.config.model_name} ...")
        
        if self.config.model_name == "virchow2":
            # 1. Virchow2 加载 (timm)
            model = timm.create_model(
                "hf_hub:paige-ai/Virchow2", 
                pretrained=True, 
                mlp_layer=timm.layers.SwiGLUPacked, 
                act_layer=torch.nn.SiLU
            )
            model.to(self.config.device)
            model.eval()
            
            # 标准 ImageNet 归一化 (直接一步到位，坚决摒弃 CenterCrop 与二次缩放)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            return model, transform

        elif self.config.model_name == "conch":
            # 2. CONCH 加载 (使用官方库)
            from conch.open_clip_custom import create_model_from_pretrained
            model, conch_transform = create_model_from_pretrained(
                model_cfg="conch_ViT-B-16", 
                checkpoint_path="hf_hub:MahmoodLab/CONCH"
            )
            model.to(self.config.device)
            model.eval()
            return model, conch_transform

        else:
            raise ValueError(f"不支持的模型: {self.config.model_name}，仅支持 'virchow2' 或 'conch'")

    def process_wsi(self, wsi_path: Path, output_dir: Path) -> Tuple[bool, str]:
        """
        处理单个 WSI 文件的完整流程 (纯串行，磁盘独占)
        """
        basename = wsi_path.stem
        start_time = time.time()
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # 获取切片基本信息
            with openslide.OpenSlide(str(wsi_path)) as slide:
                w, h = slide.dimensions
                orig_mpp = slide.properties.get('openslide.mpp-x', 'unknown')

            region_size = self.config.tile_size * self.config.virtual_downsample_factor
            
            # 1. 优先精准探测是否已有模型提取过坐标 (O(1) 路径探测，杜绝 glob("**/*") 递归导致的严重 I/O 延迟)
            coord_filename = f"{basename}_coords_downsampled{self.config.virtual_downsample_factor}x.npy"
            organ_name = output_dir.name
            coord_candidates = [
                (output_dir / coord_filename, self.config.model_name),
                (self.config.features_output_dir / "virchow2" / organ_name / coord_filename, "virchow2"),
                (self.config.features_output_dir / "conch" / organ_name / coord_filename, "conch"),
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
                    valid_tiles = self._scan_tiles(wsi_path, h, region_size)
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
            final_feats = self._extract_features(wsi_path, final_tiles, region_size)
            extract_time = time.time() - extract_start
            
            if final_feats is None:
                logger.error(f"❌ [提取失败] {basename}: 模型返回空特征")
                return False, 'extraction_failed'

            # 【核心修复】数值安全熔断，杜绝半精度下产生 NaN 污染下游训练
            if np.isnan(final_feats).any():
                logger.error(f"❌ [数值异常] {basename}: 提取特征检测到 NaN，拒绝入库！")
                return False, 'nan_detected'

            # 3. 保存特征与状态 (原子操作，单线程无锁写入)
            self._save_results(basename, final_feats, final_tiles, w, h, region_size, orig_mpp, wsi_path, output_dir)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ 完成: {basename} | 形状: {final_feats.shape} | 扫描: {scan_time:.1f}s | 提取: {extract_time:.1f}s | 总计: {elapsed:.1f}s")
            return True, 'completed'

        except Exception as e:
            logger.error(f"❌ [处理错误] {basename}: {e}")
            logger.debug(traceback.format_exc())
            return False, 'error' 

    def _scan_tiles(self, wsi_path: Path, height: int, region_size: int) -> List[Dict[str, int]]:
        """具备单层 TIFF 防爆保护与硬边界约束的自适应组织扫描"""
        with openslide.OpenSlide(str(wsi_path)) as slide:
            w, h = slide.dimensions
            
            # --- 情况 A: 包含金字塔多层结构，安全执行缩略图极速 Otsu ---
            if slide.level_count > 1:
                target_dim = 2048
                scale = max(w, h) / target_dim
                thumb_w, thumb_h = int(w / scale), int(h / scale)
                thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
                
                hsv = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2HSV)
                s_channel = hsv[:, :, 1]
                # S-Channel: 最低地板阈值设为 15，并增加上限 35 保护，避免大面积深染导致浅色组织（如黏液湖/粉刺样坏死/脂肪）被误判丢弃
                otsu_thresh, _ = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                final_thresh = min(max(otsu_thresh, 15), 35)
                _, mask = cv2.threshold(s_channel, final_thresh, 255, cv2.THRESH_BINARY)
                
                thumb_step = region_size / scale
                valid_tiles = []
                for y in range(0, h, region_size):
                    ty = int(y / scale)
                    ty_end = min(int(ty + thumb_step), thumb_h)
                    for x in range(0, w, region_size):
                        tx = int(x / scale)
                        tx_end = min(int(tx + thumb_step), thumb_w)
                        sub_mask = mask[ty:ty_end, tx:tx_end]
                        if sub_mask.size > 0 and (np.count_nonzero(sub_mask) / sub_mask.size) >= 0.15:
                            # 确保瓦片完全落在切片有效图像尺寸内，杜绝越界产生黑边半截瓦片
                            if (x + region_size <= w) and (y + region_size <= h):
                                valid_tiles.append({'x': x, 'y': y})
                return valid_tiles

        # --- 情况 B: 扁平单层 TIFF (level_count == 1)，避免 get_thumbnail 爆内存 ---
        # 回退至安全的多进程瓦片读取 (使用已有的 tile_worker_func)
        y_coords = [y for y in range(0, height, region_size) if y + region_size <= height]
        y_chunks = np.array_split(np.array(y_coords), self.config.num_workers_tiling)
        worker_args = [
            (wsi_path, chunk, 0, region_size, self.config.tile_size, self.config) 
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

    def _extract_features(self, wsi_path: Path, tiles: List[Dict[str, int]], region_size: int) -> Optional[np.ndarray]:
        """特征提取：严格遵循论文规范，Virchow2 仅保留纯 Class Token"""
        dataset = WSITileDataset(wsi_path, tiles, region_size, self.config.tile_size, transform=self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_dataloader,
            shuffle=False,
            pin_memory=True, # 加速 CPU 到 GPU 的传输
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
                    if self.config.model_name == "virchow2":
                        # Virchow2: 取 index 0 的 Class Token (1280 维)
                        output = self.model(batch_imgs)
                        feats = output[:, 0]
                    elif self.config.model_name == "conch":
                        # CONCH: 投影至跨模态潜空间的图块特征 (512 维)
                        feats = self.model.encode_image(batch_imgs, proj_contrast=True, normalize=False)
                    else:
                        raise ValueError(f"不支持的模型: {self.config.model_name}")
                    
                all_feats.append(feats.float().cpu().numpy()) # 转回 float32 保存

        if not all_feats:
            return None
        return np.vstack(all_feats)

    def _save_results(self, basename: str, feats: np.ndarray, tiles: List[Dict[str, int]], 
                     w: int, h: int, region_size: int, mpp: str, wsi_path: Path, output_dir: Path):
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
            'mpp': str(mpp)
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
                self.config.features_output_dir / "conch" / organ_name / "qc_overlay" / f"{basename}_QC.jpg",
                self.config.features_output_dir / organ_name / "qc_overlay" / f"{basename}_QC.jpg",
            ]
            found_qc = next((p for p in qc_candidates if p.is_file()), None)
            if found_qc:
                shutil.copyfile(found_qc, qc_file)
            else:
                self._save_visualization(wsi_path, tiles, qc_dir, basename, region_size)

    def _save_visualization(self, wsi_path: Path, tiles: List[Dict[str, int]], output_dir: Path, basename: str, region_size: int):
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
    parser = argparse.ArgumentParser(description="串行极速版: Virchow2 与 CONCH 双特征提取 (磁盘独占无冲突)")
    parser.add_argument('--model', type=str, default='virchow2', choices=['virchow2', 'conch'], 
                        help='选择提取特征的基础模型: virchow2 (1280维) 或 conch (512维)')
    parser.add_argument('--organs', type=str, help='按器官筛选')
    parser.add_argument('--ids', type=str, help='按 ID 筛选')
    args = parser.parse_args()

    # 1. 配置与日志
    config = AppConfig()
    config.model_name = args.model # 动态覆盖模型

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
