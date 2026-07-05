# run_stage1_v3_oop.py (第三版：深度重构 OOP + 类型提示 + 日志系统)
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
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any

import cv2
import numpy as np
import pandas as pd
import torch
import timm
import openslide
from PIL import Image, PngImagePlugin, ImageDraw
from torchvision import transforms
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
    # 路径配置 (请根据您的实际环境修改这些路径)
    wsi_input_dir: Path = Path("/mnt/e/REG_train_CLEANED")  # WSI 原始文件输入目录
    features_output_dir: Path = Path("/mnt/e/Extracted_Features_train_CLEANED_all_tiles") # 特征输出目录
    gigapath_repo_path: Path = Path("~/projects/prov-gigapath").expanduser() # GigaPath 代码库路径
    status_cache_path: Path = Path("~/projects/wsi_status_cache_train_CLEANED_all_tiles.csv").expanduser() # 状态缓存文件路径
    json_path: Path = Path("train.json") # 包含 WSI 元数据的 JSON 文件

    # 处理参数
    virtual_downsample_factor: int = 2 # 虚拟下采样倍率 (例如 2 表示长宽各缩小一半)
    tile_size: int = 256 # 图块大小 (像素)
    max_tiles_per_wsi: int = 12000 # 每个 WSI 保留的最大图块数 (超过则随机采样)
    min_tiles_per_wsi: int = 16 # 每个 WSI 最少需要的有效图块数 (少于此数则视为无效样本)
    batch_size: int = 64 # 特征提取时的批次大小 (根据显存调整)
    num_workers_dataloader: int = 15 # [优化] 给数据加载分配 14 个核 (重兵投入喂 GPU)
    num_workers_tiling: int = 5 # [优化] 给切片扫描分配 4 个核 (足够了)
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
        self.gigapath_repo_path = Path(self.gigapath_repo_path)
        self.status_cache_path = Path(self.status_cache_path)
        self.json_path = Path(self.json_path)

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
    wsi_path, y_coords_chunk, level, region_size, tile_size, cfg = args
    local_tiles = []
    slide = None
    
    try:
        # 在子进程中打开 Slide
        slide = openslide.OpenSlide(str(wsi_path))
        level_w, _ = slide.dimensions
        
        # 遍历分配给该进程的 Y 坐标块
        for y in y_coords_chunk:
            for x in range(0, level_w, region_size):
                try:
                    # 读取区域并缩放
                    large_region_pil = slide.read_region((x, y), level, (region_size, region_size)).convert("RGB")
                    tile_pil = large_region_pil.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
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

                    # 通过所有检查，保留该图块坐标
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
            # 从大图中读取特定区域
            img = slide.read_region((t['x'], t['y']), 0, (self.region_size, self.region_size)).convert("RGB")
            # 缩放到模型输入大小
            img = img.resize((self.tile_size, self.tile_size), Image.Resampling.LANCZOS)
            
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            logger.error(f"Error reading tile {t}: {e}")
            # 出错时返回全零张量，保证 batch 形状一致，避免程序崩溃
            return torch.zeros((3, 224, 224)) 

    def __del__(self):
        """析构函数，确保释放 OpenSlide 资源"""
        if self._slide:
            self._slide.close()

def worker_init_fn(worker_id):
    """DataLoader worker 初始化函数 (占位符)"""
    pass

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
                    mapping[item['id']] = organ
            logger.info(f"已加载 {len(mapping)} 个样本映射。")
        except Exception as e:
            logger.error(f"JSON读取错误: {e}")
        return mapping

    def _load_status_cache(self) -> Dict[str, str]:
        """加载之前的处理状态缓存，避免重复处理"""
        if self.config.status_cache_path.exists():
            try:
                return pd.read_csv(self.config.status_cache_path).set_index('wsi_filename')['status'].to_dict()
            except Exception:
                return {}
        return {}

    def update_status(self, wsi_filename: str, status: str):
        """
        更新处理状态并保存到 CSV 文件
        """
        try:
            if not self.config.status_cache_path.exists():
                df = pd.DataFrame(columns=['wsi_filename', 'status'])
            else:
                df = pd.read_csv(self.config.status_cache_path)
            
            if wsi_filename in df['wsi_filename'].values:
                df.loc[df['wsi_filename'] == wsi_filename, 'status'] = status
            else:
                new_row = pd.DataFrame([{'wsi_filename': wsi_filename, 'status': status}])
                df = pd.concat([df, new_row], ignore_index=True)
            
            df.to_csv(self.config.status_cache_path, index=False)
            self.status_cache[wsi_filename] = status # 同步更新内存缓存
        except Exception as e:
            logger.error(f"Failed to update status cache: {e}")

    def get_pending_files(self, organ_filter: Optional[Set[str]], id_filter: Optional[Set[str]]) -> List[Tuple[Path, Path]]:
        """
        获取待处理的文件列表
        """
        all_files = sorted(list(self.config.wsi_input_dir.glob("*.tif*")))
        pending = []

        for path in all_files:
            filename = path.name
            basename = path.stem

            # 1. ID 筛选
            if id_filter and basename not in id_filter:
                continue
            
            # 2. 获取器官信息
            organ = self.wsi_to_organ_map.get(filename, "Unknown")
            
            # 3. 器官筛选
            if organ_filter and organ not in organ_filter:
                continue

            organ_dir = self.config.features_output_dir / organ

            # 4. 状态检查 (去子目录里检查特征文件是否存在)
            is_completed = self.status_cache.get(filename) == 'completed'
            feature_file = organ_dir / f'{basename}_features_downsampled{self.config.virtual_downsample_factor}x.npy'
            
            if not (is_completed and feature_file.exists()):
                # [修改] 将 WSI 路径和 目标目录 一起加入列表
                pending.append((path, organ_dir))
        
        return pending

class WSIProcessor:
    """
    核心处理器类
    封装了切片、特征提取和结果保存的完整流水线。
    """
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = self._load_model()
        # 定义图像预处理变换 (Resize -> Crop -> Tensor -> Normalize)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        # 准备输出目录
        self.permanent_tiles_dir = self.config.features_output_dir / "WSI_Top_Tiles_Output_train_CLEANED_all_tiles"
        self.config.features_output_dir.mkdir(parents=True, exist_ok=True)
        self.permanent_tiles_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """加载 GigaPath 预训练模型"""
        logger.info("正在加载 GigaPath 模型...")
        try:
            # 确保 GigaPath 在 Python 路径中
            if str(self.config.gigapath_repo_path) not in sys.path:
                sys.path.append(str(self.config.gigapath_repo_path))
            
            model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            model.to(self.config.device)
            model.eval() # 设置为评估模式
            return model
        except Exception as e:
            logger.critical(f"加载模型失败: {e}")
            raise

    def process_wsi(self, wsi_path: Path, output_dir: Path) -> Tuple[bool, str]:
        """
        处理单个 WSI 文件的入口函数
        """
        basename = wsi_path.stem
        
        # [修改] 临时目录也放在器官文件夹下，保持整洁
        temp_tile_dir = output_dir / "temp_tiles" / basename
        
        try:
            # 清理临时目录
            if temp_tile_dir.exists():
                shutil.rmtree(temp_tile_dir, ignore_errors=True)
            temp_tile_dir.mkdir(parents=True, exist_ok=True)

            # 1. 获取基本信息 (代码不变)
            with openslide.OpenSlide(str(wsi_path)) as slide:
                w, h = slide.dimensions
                orig_mpp = slide.properties.get('openslide.mpp-x', 'unknown')

            region_size = self.config.tile_size * self.config.virtual_downsample_factor
            
            # 2. 扫描有效图块 (代码不变)
            valid_tiles = self._scan_tiles(wsi_path, h, region_size)
            
            if len(valid_tiles) < self.config.min_tiles_per_wsi:
                logger.warning(f"!!! 有效图块不足 ({len(valid_tiles)})")
                return False, 'no_tiles'

            # 3. 采样 (代码不变)
            final_tiles = self._sample_tiles(valid_tiles)
            
            # 4. 特征提取 (代码不变)
            final_feats = self._extract_features(wsi_path, final_tiles, region_size)
            
            if final_feats is None:
                return False, 'extraction_failed'

            # 5. 保存结果 [修改] 传入 output_dir
            self._save_results(basename, final_feats, final_tiles, w, h, region_size, orig_mpp, wsi_path, output_dir)
            
            return True, 'completed'

        except Exception as e:
            logger.error(f"处理出错 {basename}: {e}")
            logger.debug(traceback.format_exc())
            return False, 'error'
        finally:
            shutil.rmtree(temp_tile_dir, ignore_errors=True)

    def _scan_tiles(self, wsi_path: Path, height: int, region_size: int) -> List[Dict[str, int]]:
        """使用多进程扫描 WSI 获取有效图块坐标"""
        y_coords = range(0, height, region_size)
        y_chunks = np.array_split(np.array(y_coords), self.config.num_workers_tiling)
        
        # 准备每个 worker 的参数
        worker_args = [
            (wsi_path, chunk, 0, region_size, self.config.tile_size, self.config) 
            for chunk in y_chunks if len(chunk) > 0
        ]
        
        valid_tiles = []
        # 使用 Pool.imap_unordered 并行处理，并显示进度条
        with multiprocessing.Pool(processes=self.config.num_workers_tiling) as pool:
            for result in tqdm(pool.imap_unordered(tile_worker_func, worker_args), 
                             total=len(worker_args), desc="扫描 WSI", leave=True):
                valid_tiles.extend(result)
        
        return valid_tiles

    def _sample_tiles(self, tiles: List[Dict[str, int]]) -> List[Dict[str, int]]:
        """如果图块过多，进行随机采样以控制数据量"""
        if len(tiles) <= self.config.max_tiles_per_wsi:
            logger.info(f"保留所有 {len(tiles)} 个图块。")
            final = tiles
        else:
            logger.info(f"从 {len(tiles)} 个图块中随机抽取 {self.config.max_tiles_per_wsi} 个。")
            final = random.sample(tiles, self.config.max_tiles_per_wsi)
        
        # 按坐标排序，保证处理顺序一致性
        final.sort(key=lambda t: (t['y'], t['x']))
        return final

    def _extract_features(self, wsi_path: Path, tiles: List[Dict[str, int]], region_size: int) -> Optional[np.ndarray]:
        """使用 DataLoader 和 GPU 模型提取特征"""
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
        with torch.no_grad():
            for batch_imgs in tqdm(loader, desc="提取特征", leave=True):
                batch_imgs = batch_imgs.to(self.config.device)
                feats = self.model(batch_imgs)
                all_feats.append(feats.cpu().numpy())

        if not all_feats:
            return None
        return np.vstack(all_feats)

    def _save_results(self, basename: str, feats: np.ndarray, tiles: List[Dict[str, int]], 
                     w: int, h: int, region_size: int, mpp: str, wsi_path: Path, output_dir: Path):
        """保存特征、坐标、元数据和 QC 图到指定目录"""
        
        # 1. 保存特征和坐标 (.npy) -> 使用 output_dir
        coords = np.array([[t['x'], t['y']] for t in tiles], dtype=np.int32)
        np.save(output_dir / f'{basename}_features_downsampled{self.config.virtual_downsample_factor}x.npy', feats)
        np.save(output_dir / f'{basename}_coords_downsampled{self.config.virtual_downsample_factor}x.npy', coords)

        # 2. 保存元数据 (.json) -> 使用 output_dir
        metadata = {
            'wsi_id': basename,
            'original_width': w,
            'original_height': h,
            'num_tiles': len(tiles),
            'patch_size': region_size,
            'mpp': str(mpp)
        }
        with open(output_dir / f'{basename}_meta.json', 'w') as f:
            json.dump(metadata, f)

        # 3. 生成 QC 图 -> 使用 output_dir 下的子文件夹
        qc_dir = output_dir / "qc_overlay"
        qc_dir.mkdir(parents=True, exist_ok=True)
        self._save_visualization(wsi_path, tiles, qc_dir, basename, region_size)

    def _save_visualization(self, wsi_path: Path, tiles: List[Dict[str, int]], output_dir: Path, basename: str, region_size: int):
        """生成并保存可视化质控图：在缩略图上绘制绿色方框"""
        try:
            with openslide.OpenSlide(str(wsi_path)) as slide:
                w, h = slide.dimensions
                target_dim = 2048 # 限制缩略图最大尺寸
                downsample = max(w, h) / target_dim
                thumb_size = (int(w / downsample), int(h / downsample))
                
                thumbnail = slide.get_thumbnail(thumb_size).convert("RGB")
                draw = ImageDraw.Draw(thumbnail)
                
                scale_x = thumb_size[0] / w
                scale_y = thumb_size[1] / h
                
                for t in tiles:
                    x, y = t['x'], t['y']
                    # 坐标转换：原图 -> 缩略图
                    x_thumb = int(x * scale_x)
                    y_thumb = int(y * scale_y)
                    w_thumb = int(region_size * scale_x)
                    h_thumb = int(region_size * scale_y)
                    
                    draw.rectangle([x_thumb, y_thumb, x_thumb + w_thumb, y_thumb + h_thumb], outline="green", width=2)
                
                thumbnail.save(output_dir / f"{basename}_QC.jpg", "JPEG", quality=80)
        except Exception as e:
            logger.warning(f"可视化失败 {basename}: {e}")

# --- 5. 核心线程逻辑 ---

def producer_scan_worker(manager: WSIManager, processor: WSIProcessor, 
                        files: List[Tuple[Path, Path]], task_queue: queue.Queue):
    """
    [生产者线程]
    负责 CPU 密集型任务：扫描 WSI 获取有效图块坐标。
    """
    logger.info("🔪 [生产者] 扫描线程启动...")
    
    for wsi_path, organ_dir in files:
        try:
            logger.info(f"🔍 [扫描中] {wsi_path.name} ...")
            
            # 1. 确保目录存在
            organ_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. 获取基本信息 (轻量)
            with openslide.OpenSlide(str(wsi_path)) as slide:
                w, h = slide.dimensions
                orig_mpp = slide.properties.get('openslide.mpp-x', 'unknown')
            
            region_size = processor.config.tile_size * processor.config.virtual_downsample_factor
            
            # 3. 扫描图块 (CPU 密集)
            start_time = time.time()
            valid_tiles = processor._scan_tiles(wsi_path, h, region_size)
            
            if len(valid_tiles) < processor.config.min_tiles_per_wsi:
                logger.warning(f"⏩ [跳过] {wsi_path.name}: 图块不足 ({len(valid_tiles)})")
                manager.update_status(wsi_path.name, 'no_tiles')
                continue

            # 4. 采样 (快速)
            final_tiles = processor._sample_tiles(valid_tiles)
            
            # 5. 打包数据放入队列
            # 数据包: (path, feats, tiles, w, h, region_size, mpp, organ_dir)
            # 注意: 这里我们只放图块坐标，特征提取在消费者线程做
            task_item = {
                'wsi_path': wsi_path,
                'organ_dir': organ_dir,
                'basename': wsi_path.stem,
                'tiles': final_tiles,
                'w': w, 'h': h,
                'region_size': region_size,
                'mpp': orig_mpp,
                'scan_time': time.time() - start_time
            }
            
            # 阻塞放入队列 (如果 GPU 慢，这里会暂停，防止内存爆掉)
            task_queue.put(task_item)
            logger.info(f"📦 [已入队] {wsi_path.name} (图块: {len(final_tiles)})")
            
        except Exception as e:
            logger.error(f"❌ [扫描失败] {wsi_path.name}: {e}")
            manager.update_status(wsi_path.name, 'scan_error')

    # 发送结束信号
    task_queue.put(None)
    logger.info("🛑 [生产者] 所有文件扫描完毕，发送结束信号。")

def consumer_gpu_worker(manager: WSIManager, processor: WSIProcessor, task_queue: queue.Queue):
    """
    [消费者线程]
    负责 GPU 密集型任务：特征提取，以及结果保存。
    """
    logger.info("🚀 [消费者] GPU 提取线程启动，等待任务...")
    
    while True:
        item = task_queue.get()
        if item is None:
            logger.info("🛑 [消费者] 收到结束信号，停止工作。")
            task_queue.task_done()
            break
            
        wsi_path = item['wsi_path']
        basename = item['basename']
        logger.info(f"⚙️ [提取中] {basename} (队列剩余: {task_queue.qsize()})")
        
        try:
            # 1. 特征提取 (GPU 密集)
            start_time = time.time()
            final_feats = processor._extract_features(wsi_path, item['tiles'], item['region_size'])
            
            if final_feats is None:
                logger.error(f"❌ [提取失败] {basename}: 模型返回空")
                manager.update_status(wsi_path.name, 'extraction_failed')
            else:
                # 2. 保存结果 (I/O)
                processor._save_results(
                    basename, final_feats, item['tiles'], 
                    item['w'], item['h'], item['region_size'], 
                    item['mpp'], wsi_path, item['organ_dir']
                )
                
                elapsed = time.time() - start_time
                scan_time = item['scan_time']
                
                # 记录成功
                manager.update_status(wsi_path.name, 'completed')
                logger.info(f"✅ 完成: {basename} | 形状: {final_feats.shape} | 扫描: {scan_time:.1f}s | 提取: {elapsed:.1f}s")
                
        except Exception as e:
            logger.error(f"❌ [提取/保存错误] {basename}: {e}")
            logger.debug(traceback.format_exc())
            manager.update_status(wsi_path.name, 'process_error')
        finally:
            task_queue.task_done()

# --- 6. 主程序 (V4) ---

def main():
    parser = argparse.ArgumentParser(description="V4 流水线版: 异步全双工特征提取")
    parser.add_argument('--organs', type=str, help='按器官筛选')
    parser.add_argument('--ids', type=str, help='按 ID 筛选')
    args = parser.parse_args()

    # 1. 配置与日志
    config = AppConfig()
    global logger
    logger = setup_logging(config.features_output_dir)
    logger.info("--- V4 异步流水线极速版 (CPU/GPU 并行) 启动 ---")
    
    try:
        # 2. 初始化单例
        manager = WSIManager(config)
        processor = WSIProcessor(config) # 这里会加载模型进 GPU

        # 3. 获取任务列表
        organ_filter = {o.strip() for o in args.organs.split(',')} if args.organs else None
        id_filter = {i.strip() for i in args.ids.split(',')} if args.ids else None
        
        files_to_process = manager.get_pending_files(organ_filter, id_filter)
        logger.info(f"📋 待处理任务: {len(files_to_process)} 个文件")
        
        if not files_to_process:
            logger.info("没有任务需要处理，退出。")
            return

        # 4. 启动流水线
        # 创建限制大小的队列，防止 Scanner 跑太快把内存吃光
        # [优化] 增大缓冲区到 20，给系统更多弹性
        task_queue = queue.Queue(maxsize=20) 

        # 创建线程
        producer = threading.Thread(target=producer_scan_worker, 
                                  args=(manager, processor, files_to_process, task_queue),
                                  name="ScannerThread")
        
        consumer = threading.Thread(target=consumer_gpu_worker, 
                                  args=(manager, processor, task_queue),
                                  name="ExtractorThread")

        start_time_total = time.time()
        
        # 启动
        logger.info("🔥 启动双线程流水线...")
        producer.start()
        consumer.start()
        
        # 等待结束
        producer.join() # 等待生产者扫完所有文件
        consumer.join() # 等待消费者处理完队列里的剩余任务
        
        total_time = time.time() - start_time_total
        logger.info(f"🏁 全部任务完成！总耗时: {total_time:.1f}s")

    except KeyboardInterrupt:
        logger.warning("⚠️ 用户中断 (Ctrl+C)，正在停止...")
        # 线程清理比较麻烦，这里直接退出即可，OS 会回收资源
        sys.exit(1)
    except Exception as e:
        logger.critical(f"系统严重错误: {e}")
        logger.debug(traceback.format_exc())

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    import queue # 这里导入以便在 main里使用，或者放到顶部
    import threading
    main()
