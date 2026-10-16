import os
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg') # 【新增】强制使用非交互式后端，防止 WSL 下绘图为空
import matplotlib.pyplot as plt
import warnings # 记得在文件头部导入这个
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.cuda.amp import GradScaler # 【新增】混合精度训练
from pathlib import Path

# --- 1. 配置与超参数 ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- 2. ASL Loss (解决长尾多标签的核心) ---
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        # 强制转为 float32 计算，防止半精度下对数/指数溢出
        x = x.float()
        y = y.float()
        x_sigmoid = torch.sigmoid(x)
        # 为正例保留微小余量，防止 log(0) 导致反向传播梯度截断为 0
        xs_pos = x_sigmoid.clamp(min=self.eps, max=1.0 - self.eps)
        xs_neg = (1.0 - x_sigmoid).clamp(min=self.eps, max=1.0 - self.eps)

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = y * torch.log(xs_pos)
        los_neg = (1.0 - y) * torch.log(xs_neg)
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1.0 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma).detach()
            loss = loss * one_sided_w

        return -loss.sum()

# --- 3. 模型定义 (MLP Head版) ---
class PredictionHead(nn.Module):
    """
    【新增】MLP 预测头：Linear -> ReLU -> Dropout -> Linear
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class MultiBranchAttentionMIL(nn.Module):
    """
    改进版 CLAM-MB 架构:
    1. 22 个主类别独立注意力分支 (A_k)
    2. NST 辅助任务绑定 NST 专属聚合特征 M_nst (index 0)
    3. DCIS 辅助任务绑定 DCIS 专属聚合特征 M_dcis (index 1)
    """
    def __init__(self, n_classes, input_dim=1280, hidden_dim=512, dropout=0.25, idx_nst=0, idx_dcis=1):
        super(MultiBranchAttentionMIL, self).__init__()
        self.n_classes = n_classes
        self.idx_nst = idx_nst
        self.idx_dcis = idx_dcis
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, 256), nn.Sigmoid())
        
        self.attention_weights = nn.Linear(256, n_classes)
        self.classifiers = nn.Linear(hidden_dim, n_classes)
        
        # 辅助多任务头
        self.head_nst_tubule = PredictionHead(hidden_dim, 3)
        self.head_nst_nuclear = PredictionHead(hidden_dim, 3)
        self.head_nst_mitoses = PredictionHead(hidden_dim, 3)
        self.head_nst_overall = PredictionHead(hidden_dim, 3)
        
        self.head_dcis_grade = PredictionHead(hidden_dim, 3)
        self.head_dcis_necrosis = PredictionHead(hidden_dim, 3)
        self.head_dcis_types = PredictionHead(hidden_dim, 3)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(0) # [1, N, input_dim] -> [N, input_dim]
            
        H = self.feature_extractor(x) # [N, hidden_dim]
        
        # 门控注意力矩阵: [N, 256] -> [N, n_classes]
        A = self.attention_weights(self.attention_V(H) * self.attention_U(H))
        A = torch.transpose(A, 1, 0) # [n_classes, N]
        A = F.softmax(A, dim=1)      # [n_classes, N]
        
        # 类别特异性聚合: [n_classes, hidden_dim]
        M = torch.mm(A, H)
        
        # 主任务预测
        logits_main = (M * self.classifiers.weight).sum(dim=1) + self.classifiers.bias
        logits_main = logits_main.unsqueeze(0) # [1, n_classes]
        
        # 【核心改进】类特异性特征解耦：
        # NST 子任务直接使用 NST 注意力分支聚合特征；DCIS 子任务使用 DCIS 分支聚合特征
        M_nst = M[self.idx_nst:self.idx_nst+1]   # [1, hidden_dim]
        M_dcis = M[self.idx_dcis:self.idx_dcis+1] # [1, hidden_dim]
        
        outputs = {
            'main': logits_main,
            'nst_tubule': self.head_nst_tubule(M_nst),
            'nst_nuclear': self.head_nst_nuclear(M_nst),
            'nst_mitoses': self.head_nst_mitoses(M_nst),
            'nst_overall': self.head_nst_overall(M_nst),
            'dcis_grade': self.head_dcis_grade(M_dcis),
            'dcis_necrosis': self.head_dcis_necrosis(M_dcis),
            'dcis_types': self.head_dcis_types(M_dcis)
        }
        return outputs, A, M

# --- 4. Dataset ---
class BreastDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_cols: list, backbone: str = 'virchow2', training: bool = True):
        # stage3 内部防御
        clean_bb = backbone.lower().replace('-', '_')
        if clean_bb == 'hoptimus1':
            clean_bb = 'h_optimus_1'
        self.feature_col = f"{clean_bb}_feature_path"

        if self.feature_col not in df.columns:
            raise KeyError(
                f"❌ 未在数据表中找到骨干网络 '{backbone}' 对应的特征列 '{self.feature_col}'！"
                f"当前可用特征列: {[c for c in df.columns if 'feature' in c]}"
            )
        
        # 过滤有效路径
        self.df = df[df[self.feature_col].notna()].reset_index(drop=True)
        self.label_cols = label_cols
        self.training = training
        
        self.nst_cols = ['nst_grade_tubule', 'nst_grade_nuclear', 'nst_grade_mitoses', 'nst_grade_overall']
        self.dcis_grade_col = 'dcis_grade'
        self.dcis_necrosis_col = 'dcis_necrosis'
        self.dcis_type_cols = ['dcis_type_solid', 'dcis_type_cribriform', 'dcis_type_micropapillary']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_path = row[self.feature_col]
        
        features = np.load(feat_path).astype(np.float32).copy()
        
        # =========================================================================
        # 🎯 22 维全标签形态感知分流：精准拦截所有极微局灶病变
        # =========================================================================
        # 1. 独立极微局灶病变
        is_micro_lesion = (
            row.get('label_microcalcification', 0) == 1 or
            row.get('label_micro_invasive', 0) == 1 or
            row.get('label_fea', 0) == 1 or
            row.get('label_adh', 0) == 1 or
            row.get('label_lcis', 0) == 1 or
            row.get('label_micropapillary', 0) == 1
        )

        # 2. 伴随性隐匿病灶 (大浸润癌背景下藏着的微小原位癌)
        is_companion_dcis = (
            row.get('label_dcis', 0) == 1 and 
            (row.get('label_invasive_nst', 0) == 1 or row.get('label_invasive_lobular', 0) == 1)
        )

        is_focal_risk = is_micro_lesion or is_companion_dcis

        # =========================================================================
        # 采样执行策略
        # =========================================================================
        if self.training:
            num_patches = features.shape[0]
            
            if is_focal_risk:
                # 【极微局灶组】：100% 全量保全，绝不冒丢失关键瓦片的风险
                pass
            else:
                # 【弥漫主体组】：覆盖穿刺标本的绝大部分面积 (上限 3500) + 温和 85%~95% 扰动
                max_bag_size = 3500
                keep_ratio = np.random.uniform(0.85, 0.95)
                target_num = min(int(num_patches * keep_ratio), max_bag_size)
                
                if num_patches > target_num:
                    indices = np.random.choice(num_patches, target_num, replace=False)
                    indices.sort()
                    features = features[indices]

            features = torch.from_numpy(features).float()
            
            # 特征微扰与缩放
            noise = torch.randn_like(features) * np.random.uniform(0.0, 0.01)
            features = (features + noise) * np.random.uniform(0.98, 1.02)
        else:
            features = torch.from_numpy(features).float()

        labels = torch.from_numpy(row[self.label_cols].fillna(0).values.astype(np.float32)).float()
        
        def _safe_int(val, default=-1):
            return default if pd.isna(val) else int(val)

        aux_targets = {}
        aux_targets['nst_grades'] = torch.tensor([
            _safe_int(row.get(c, -1)) - 1 if _safe_int(row.get(c, -1)) > 0 else -1 
            for c in self.nst_cols
        ], dtype=torch.long)
        
        d_grade = _safe_int(row.get(self.dcis_grade_col, -1))
        aux_targets['dcis_grade'] = torch.tensor(d_grade - 1 if d_grade > 0 else -1, dtype=torch.long)
        
        d_necrosis = _safe_int(row.get(self.dcis_necrosis_col, -1))
        aux_targets['dcis_necrosis'] = torch.tensor(d_necrosis, dtype=torch.long)
        
        aux_targets['dcis_types'] = torch.from_numpy(
            row[self.dcis_type_cols].fillna(0).values.astype(np.float32)
        )

        return features, labels, aux_targets

def get_strategic_sampler(dataset):
    # 防护：空值赋 0 并转 float32，避免隐式空值导致除零或 NaN
    targets = dataset.df[dataset.label_cols].fillna(0).values.astype(np.float32)
    class_sample_count = targets.sum(axis=0)
    class_sample_count = np.maximum(class_sample_count, 1)
    
    total_samples = len(dataset)
    weights_per_class = np.sqrt(total_samples / (class_sample_count + 5.0))
    weights_per_class = np.clip(weights_per_class / weights_per_class.min(), 1.0, 15.0)
    
    print("\n--- [采样器策略核查 (部分典型类别)] ---")
    monitor_cols = ['label_invasive_nst', 'label_mucinous', 'label_lymphoma', 'label_micro_invasive']
    for col_name in monitor_cols:
        if col_name in dataset.label_cols:
            idx = dataset.label_cols.index(col_name)
            print(f"类别 {col_name:<25} (样本数={int(class_sample_count[idx]):>3}): 采样倍率 = {weights_per_class[idx]:.2f}x")
    
    samples_weight = np.zeros(len(dataset))
    for i in range(len(dataset)):
        label_indices = np.where(targets[i] >= 0.5)[0]
        if len(label_indices) > 0:
            samples_weight[i] = max(weights_per_class[label_indices])
        else:
            samples_weight[i] = 1.0
            
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

# --- 5. 训练与验证 (训练函数) ---
def train_one_epoch(model, loader, criterion_main, optimizer, device, label_cols, scaler=None):
    model.train()
    tracker = {'loss_total': 0, 'loss_main': 0, 'loss_nst': 0, 'loss_dcis': 0}
    
    idx_dcis = label_cols.index('label_dcis') if 'label_dcis' in label_cols else -1
    criterion_aux_ce = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_aux_bce = nn.BCEWithLogitsLoss(reduction='none')

    # 设备安全校验：优先在 3090 上开启 bfloat16，彻底杜绝 NaN
    is_cuda = torch.cuda.is_available() and "cuda" in str(device)
    has_bf16 = is_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if has_bf16 else torch.float16
    device_type = 'cuda' if is_cuda else 'cpu'

    for features, labels, aux_targets in tqdm(loader, desc="Training", leave=False):
        features, labels = features.to(device), labels.to(device)
        nst_grades = aux_targets['nst_grades'].to(device)
        dcis_grade = aux_targets['dcis_grade'].to(device)
        dcis_necrosis = aux_targets['dcis_necrosis'].to(device)
        dcis_types = aux_targets['dcis_types'].to(device)

        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=is_cuda):
            outputs, _, _ = model(features)
            loss_main = criterion_main(outputs['main'], labels)
            
            # NST 掩码 Loss (各有效子分支求平均，杜绝失衡)
            nst_losses = []
            if (nst_grades[:, 0] != -1).any(): nst_losses.append(criterion_aux_ce(outputs['nst_tubule'], nst_grades[:, 0]))
            if (nst_grades[:, 1] != -1).any(): nst_losses.append(criterion_aux_ce(outputs['nst_nuclear'], nst_grades[:, 1]))
            if (nst_grades[:, 2] != -1).any(): nst_losses.append(criterion_aux_ce(outputs['nst_mitoses'], nst_grades[:, 2]))
            if (nst_grades[:, 3] != -1).any(): nst_losses.append(criterion_aux_ce(outputs['nst_overall'], nst_grades[:, 3]))
            loss_nst = torch.stack(nst_losses).mean() if nst_losses else torch.tensor(0.0, device=device)
            
            # DCIS 掩码 Loss
            dcis_losses = []
            if (dcis_grade != -1).any(): dcis_losses.append(criterion_aux_ce(outputs['dcis_grade'], dcis_grade))
            if (dcis_necrosis != -1).any(): dcis_losses.append(criterion_aux_ce(outputs['dcis_necrosis'], dcis_necrosis))
            if idx_dcis != -1 and labels[:, idx_dcis].sum() > 0:
                raw_bce = criterion_aux_bce(outputs['dcis_types'], dcis_types)
                dcis_losses.append(raw_bce.mean()) # 3 个亚型求平均
            loss_dcis = torch.stack(dcis_losses).mean() if dcis_losses else torch.tensor(0.0, device=device)

            # 辅助损失系数设为 0.25，保障主任务占绝对主导
            loss_total = loss_main + 0.25 * (loss_nst + loss_dcis)

        # 梯度回传
        if scaler is not None and amp_dtype == torch.float16:
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        tracker['loss_total'] += loss_total.item()
        tracker['loss_main'] += loss_main.item()
        tracker['loss_nst'] += loss_nst.item()
        tracker['loss_dcis'] += loss_dcis.item()

    for k in tracker:
        tracker[k] /= max(len(loader), 1)
        
    return tracker
    
# --- 5. 训练与验证 (验证函数) ---
def validate(model, loader, device, label_names, return_attention=False):
    model.eval()
    all_targets, all_probs = [], []
    all_attentions = []
    sub_task_preds = {'nst_n': [], 'nst_t': [], 'nst_m': [], 'nst_o': [], 'dcis_g': [], 'dcis_n': [], 'dcis_t': []}
    sub_task_targets = {'nst_n': [], 'nst_t': [], 'nst_m': [], 'nst_o': [], 'dcis_g': [], 'dcis_n': [], 'dcis_t': []}
    
    is_cuda = torch.cuda.is_available() and "cuda" in str(device)
    amp_dtype = torch.bfloat16 if (is_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    device_type = 'cuda' if is_cuda else 'cpu'

    with torch.no_grad():
        for features, labels, aux in loader:
            features, labels = features.to(device), labels.to(device)
            
            # 开启混合精度评估，与训练精度对齐并节省显存
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=is_cuda):
                outputs, A, _ = model(features)
            
            logits = outputs['main']
            probs = torch.sigmoid(logits)
            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            if return_attention:
                all_attentions.append(A.squeeze(0).cpu().numpy() if A.dim() == 3 else A.cpu().numpy())
            
            # NST 子任务
            sub_task_preds['nst_t'].append(outputs['nst_tubule'].argmax(dim=1).cpu().numpy())
            sub_task_targets['nst_t'].append(aux['nst_grades'][:, 0].cpu().numpy())
            sub_task_preds['nst_n'].append(outputs['nst_nuclear'].argmax(dim=1).cpu().numpy())
            sub_task_targets['nst_n'].append(aux['nst_grades'][:, 1].cpu().numpy())
            sub_task_preds['nst_m'].append(outputs['nst_mitoses'].argmax(dim=1).cpu().numpy())
            sub_task_targets['nst_m'].append(aux['nst_grades'][:, 2].cpu().numpy())
            sub_task_preds['nst_o'].append(outputs['nst_overall'].argmax(dim=1).cpu().numpy())
            sub_task_targets['nst_o'].append(aux['nst_grades'][:, 3].cpu().numpy())

            # DCIS 子任务
            sub_task_preds['dcis_g'].append(outputs['dcis_grade'].argmax(dim=1).cpu().numpy())
            sub_task_targets['dcis_g'].append(aux['dcis_grade'].cpu().numpy())
            sub_task_preds['dcis_n'].append(outputs['dcis_necrosis'].argmax(dim=1).cpu().numpy())
            sub_task_targets['dcis_n'].append(aux['dcis_necrosis'].cpu().numpy())
            
            if aux['dcis_grade'].item() != -1 or (labels[0, label_names.index('label_dcis')].item() == 1 if 'label_dcis' in label_names else False):
                sub_task_preds['dcis_t'].append((torch.sigmoid(outputs['dcis_types']) > 0.5).int().cpu().numpy())
                sub_task_targets['dcis_t'].append(aux['dcis_types'].cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    
    def calc_masked_f1(preds, targs):
        preds = np.concatenate(preds)
        targs = np.concatenate(targs)
        if len(targs.shape) > 1: 
            return f1_score(targs.astype(int), preds.astype(int), average='macro', zero_division=0)
        valid_mask = (targs != -1)
        if valid_mask.sum() == 0:
            return 0.0
        return f1_score(targs[valid_mask], preds[valid_mask], average='macro', zero_division=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid_pos_mask = (all_targets.sum(axis=0) > 0)
        if valid_pos_mask.sum() > 0:
            val_map = average_precision_score(all_targets[:, valid_pos_mask], all_probs[:, valid_pos_mask], average='macro')
            val_preds = (all_probs[:, valid_pos_mask] > 0.5).astype(int)
            val_f1 = f1_score(all_targets[:, valid_pos_mask], val_preds, average='macro', zero_division=0)
        else:
            val_map = 0.0
            val_f1 = 0.0
        
        sub_metrics = {
            'nst_tubule': calc_masked_f1(sub_task_preds['nst_t'], sub_task_targets['nst_t']),
            'nst_nuclear': calc_masked_f1(sub_task_preds['nst_n'], sub_task_targets['nst_n']),
            'nst_mitoses': calc_masked_f1(sub_task_preds['nst_m'], sub_task_targets['nst_m']),
            'nst_overall': calc_masked_f1(sub_task_preds['nst_o'], sub_task_targets['nst_o']),
            'dcis_grade': calc_masked_f1(sub_task_preds['dcis_g'], sub_task_targets['dcis_g']),
            'dcis_necrosis': calc_masked_f1(sub_task_preds['dcis_n'], sub_task_targets['dcis_n']),
            'dcis_type': calc_masked_f1(sub_task_preds['dcis_t'], sub_task_targets['dcis_t']) if len(sub_task_preds['dcis_t']) > 0 else 0.0
        }

    if return_attention:
        return val_map, val_f1, all_targets, all_probs, sub_metrics, all_attentions
    return val_map, val_f1, all_targets, all_probs, sub_metrics

def find_optimal_thresholds(targets, probs, label_names):
    print("\n--- 寻找最佳 F1 阈值 ---")
    best_thresholds = []
    n_classes = targets.shape[1]
    
    for i in range(n_classes):
        best_f1 = 0.0
        best_th = 0.5
        y_true = targets[:, i]
        y_score = probs[:, i]
        
        if y_true.sum() == 0:
            best_thresholds.append(0.5)
            continue

        # 【优化】下界放宽至 0.02，步长缩细至 0.02，精确捕获长尾类的低概率激活
        for th in np.arange(0.02, 0.95, 0.02):
            y_pred = (y_score > th).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = th
        
        # 若仍无法命中（模型输出全低于 0.02），针对罕见类赋予 0.15 的探索性安全阈值，而非 0.50
        if best_f1 == 0.0:
            best_th = 0.15
            
        best_thresholds.append(round(float(best_th), 4))
        print(f"Class {label_names[i]:<25}: Best Th={best_th:.2f}, F1={best_f1:.4f}")
        
    return best_thresholds

# --- 6. 主程序 (修改版：自动运行 5 折) ---

def run_fold(fold_idx, args, df, label_cols, n_classes):
    print(f"\n{'='*20} 开始训练 Fold {fold_idx} [{args.backbone.upper()}] {'='*20}")
    
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    log_dir = os.path.join(args.save_dir, 'logs', f'fold{fold_idx}')
    plot_dir = os.path.join(args.save_dir, 'plots')
    meta_dir = os.path.join(args.save_dir, 'metadata')
    for d in [ckpt_dir, log_dir, plot_dir, meta_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 【核心优化】直接切片 DataFrame，彻底废弃 temp_train_fold.csv 磁盘擦写
    train_df = df[df['fold'] != fold_idx].reset_index(drop=True)
    val_df = df[df['fold'] == fold_idx].reset_index(drop=True)
    
    print(f"Fold {fold_idx}: Train={len(train_df)} 例, Val={len(val_df)} 例")
    if len(val_df) == 0: return

    writer = SummaryWriter(log_dir=log_dir)
    history = {'loss': [], 'map': [], 'f1': [], 'lr': []}

    train_dataset = BreastDataset(train_df, label_cols, backbone=args.backbone, training=True)
    val_dataset = BreastDataset(val_df, label_cols, backbone=args.backbone, training=False)
    sampler = get_strategic_sampler(train_dataset)
    
    use_persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=1, sampler=sampler, 
        num_workers=args.num_workers, pin_memory=True, 
        persistent_workers=use_persistent, worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True, 
        persistent_workers=use_persistent, worker_init_fn=seed_worker
    )
    
    # 动态匹配模型输入维度
    if args.backbone == 'virchow2':
        input_dim = 1280
    elif args.backbone == 'conch':
        input_dim = 512
    elif args.backbone in ["conch1_5", "conch_1_5"]:
        input_dim = 1024
    elif args.backbone in ['h_optimus_1', 'h-optimus-1', 'hoptimus1']:
        input_dim = 1536
    elif args.backbone == 'gigapath':       
        input_dim = 1536
    else:
        raise ValueError(f"未知的 backbone: {args.backbone}")
    idx_nst = label_cols.index('label_invasive_nst') if 'label_invasive_nst' in label_cols else 0
    idx_dcis = label_cols.index('label_dcis') if 'label_dcis' in label_cols else 1
    model = MultiBranchAttentionMIL(n_classes=n_classes, input_dim=input_dim, idx_nst=idx_nst, idx_dcis=idx_dcis).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scaler = GradScaler() if (torch.cuda.is_available() and not has_bf16) else None
    
    best_score = -1.0  # 确保第 1 轮必定保存有效 checkpoint
    best_epoch = -1
    best_map = -1.0        # 【新增】
    best_map_epoch = -1    # 【新增】
    
    # 4. 训练循环
    for epoch in range(args.epochs):
        # 1. 训练
        loss_dict = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, label_cols, scaler)
        train_loss = loss_dict['loss_total'] # 取总 Loss 用于显示
        
        # 2. 验证 (接收 sub_metrics)
        val_map, val_f1, val_targets, val_probs, sub_metrics = validate(model, val_loader, DEVICE, label_cols)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # --- 1. TensorBoard 记录 (Loss & Metrics) ---
        writer.add_scalar('Train/Loss_Total', train_loss, epoch)
        writer.add_scalar('Train/Loss_Main', loss_dict['loss_main'], epoch)
        writer.add_scalar('Train/Loss_NST', loss_dict['loss_nst'], epoch)
        writer.add_scalar('Train/Loss_DCIS', loss_dict['loss_dcis'], epoch)
        
        writer.add_scalar('Val/mAP', val_map, epoch)
        writer.add_scalar('Val/F1_Macro', val_f1, epoch)

        # 记录关键子任务指标
        writer.add_scalar('Val/F1_NST_Overall', sub_metrics['nst_overall'], epoch)
        writer.add_scalar('Val/F1_NST_Nuclear', sub_metrics['nst_nuclear'], epoch)
        writer.add_scalar('Val/F1_DCIS_Grade', sub_metrics['dcis_grade'], epoch)

        # 记录所有子任务到 TensorBoard
        writer.add_scalar('SubTask/NST_Overall', sub_metrics['nst_overall'], epoch)
        writer.add_scalar('SubTask/NST_Tubule', sub_metrics['nst_tubule'], epoch)
        writer.add_scalar('SubTask/NST_Nuclear', sub_metrics['nst_nuclear'], epoch)
        writer.add_scalar('SubTask/NST_Mitoses', sub_metrics['nst_mitoses'], epoch)
        writer.add_scalar('SubTask/DCIS_Grade', sub_metrics['dcis_grade'], epoch)
        writer.add_scalar('SubTask/DCIS_Necrosis', sub_metrics['dcis_necrosis'], epoch)
        writer.add_scalar('SubTask/DCIS_Type', sub_metrics['dcis_type'], epoch)
        
        # --- 2. 关键任务监控 (打印日志 + History记录) ---
        val_preds = (val_probs > 0.5).astype(int)
        
        # 1. 标题放最前
        print(f"\n{'='*15} [Fold {fold_idx} | Ep {epoch+1}/{args.epochs}] {'='*15}")
        
        # 2. 打印主任务 Loss 和 Score
        current_score = 0.7 * val_map + 0.3 * val_f1
        print(f"Total Loss: {train_loss:.4f} (Main:{loss_dict['loss_main']:.3f} NST:{loss_dict['loss_nst']:.3f} DCIS:{loss_dict['loss_dcis']:.3f})")
        print(f"Metrics   : mAP: {val_map:.4f} | F1: {val_f1:.4f} | Score: {current_score:.4f}")
        
        # 3. 打印关键类别 F1 (完整监控长尾与罕见类)
        monitor_targets = [
            'label_invasive_nst', 'label_dcis', 'label_mucinous', 
            'label_micro_invasive', 'label_micropapillary', 'label_lymphoma', 'label_fea'
        ]
        print("-" * 65)
        print(f"{'Class':<25} | {'F1':<8} | {'True/Pred'}")
        print("-" * 65)
        for col in monitor_targets:
            if col in label_cols:
                idx = label_cols.index(col)
                f1 = f1_score(val_targets[:, idx], val_preds[:, idx], zero_division=0)
                pred_pos = val_preds[:, idx].sum()
                true_pos = val_targets[:, idx].sum()
                print(f"{col:<25} | {f1:.4f}   | {int(true_pos)}/{int(pred_pos)}")
                
                short_name = col.replace('label_', '')
                if short_name in ['mucinous', 'micro_invasive', 'micropapillary', 'lymphoma', 'fea']:
                     if f'rare_{short_name}' not in history: history[f'rare_{short_name}'] = []
                     history[f'rare_{short_name}'].append(f1)

        # 4. 打印子任务
        print("-" * 65)
        print(f"{'NST Sub-tasks':<30} | {'DCIS Sub-tasks':<30}")
        print("-" * 65)
        print(f"Overall: {sub_metrics['nst_overall']:.4f}{' '*14} | Grade   : {sub_metrics['dcis_grade']:.4f}")
        print(f"Nuclear: {sub_metrics['nst_nuclear']:.4f}{' '*14} | Necrosis: {sub_metrics['dcis_necrosis']:.4f}")
        print(f"Tub/Mit: {sub_metrics['nst_tubule']:.4f} / {sub_metrics['nst_mitoses']:.4f}{' '*7} | Type    : {sub_metrics['dcis_type']:.4f}")
        print("-" * 65)
        
        history['loss'].append(train_loss)
        history['map'].append(val_map)
        history['f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        # 保存最佳模型
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            save_path = os.path.join(ckpt_dir, f'best_model_fold{fold_idx}.pth')
            torch.save(model.state_dict(), save_path)
            print(f">>> 🏆 最佳模型已保存! (新高分: {best_score:.4f})")

        # 【新增】mAP 轨独立保存
        if val_map > best_map:
            best_map = val_map
            best_map_epoch = epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'best_model_map_fold{fold_idx}.pth'))
            print(f">>> 🎯 [mAP 轨] 最佳模型已保存! (最高 mAP: {best_map:.4f})")
        
        # 【新增】保存每一轮的 latest 模型 (防止中断)
        last_path = os.path.join(ckpt_dir, f'last_model_fold{fold_idx}.pth')
        torch.save(model.state_dict(), last_path)
        
    print(f">>> Fold {fold_idx} 结束. 最佳轮次: {best_epoch+1}, 最高分: {best_score:.4f}")
    
    # 5. 收尾工作
    writer.close()
    
    # 【修改】图片保存到 plots 文件夹
    plot_save_path = os.path.join(plot_dir, f'training_curves_fold{fold_idx}.png')
    plot_history(history, plot_save_path)
    
    # 6. 阈值搜索与 OOF 预测结果固化 (供 Stage 4 融合直读)
    print(f"正在计算 Fold {fold_idx} 的最佳阈值与 OOF 预测结果...")
    model_path = os.path.join(ckpt_dir, f'best_model_fold{fold_idx}.pth')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    
    # 单次遍历同时获取 targets, probs 与 attentions
    _, _, val_targets, val_probs, _, val_attentions = validate(
        model, val_loader, DEVICE, label_cols, return_attention=True
    )
    best_thresholds = find_optimal_thresholds(val_targets, val_probs, label_cols)
    
    # 固化阈值文件
    th_save_path = os.path.join(meta_dir, f'thresholds_fold{fold_idx}.txt')
    with open(th_save_path, 'w') as f:
        for idx, th in enumerate(best_thresholds):
            f.write(f"{label_cols[idx]},{th:.4f}\n")
    print(f"阈值已保存至 {th_save_path}")

    # 【修复】优先提取纯净的 stem 主键，杜绝带 .tiff 后缀污染热力图坐标查找
    if 'wsi_id' in val_dataset.df.columns:
        valid_wsi_ids = [Path(str(x)).stem for x in val_dataset.df['wsi_id'].values]
    elif 'id' in val_dataset.df.columns:
        valid_wsi_ids = [Path(str(x)).stem for x in val_dataset.df['id'].values]
    else:
        valid_wsi_ids = [f"slide_{i}" for i in range(len(val_probs))]

    # 建立 ID 到 Attention 矩阵的字典映射: {slide_id: A_matrix}
    attention_dict = {wsi_id: att for wsi_id, att in zip(valid_wsi_ids, val_attentions)}

    # 统一存为 .pt 文件，封装概率、真值、ID、阈值与注意力字典
    oof_save_path = os.path.join(meta_dir, f'oof_predictions_fold{fold_idx}.pt')
    torch.save({
        'probs': val_probs,
        'targets': val_targets,
        'wsi_ids': valid_wsi_ids,
        'thresholds': np.array(best_thresholds, dtype=np.float32),
        'attentions': attention_dict
    }, oof_save_path)
    print(f"✅ OOF 预测、切片注意力字典与最佳阈值已保存至 {oof_save_path}")

    # 【新增】固化 mAP 轨的 OOF 与阈值
    if best_map_epoch == best_epoch:
        import shutil
        shutil.copyfile(th_save_path, os.path.join(meta_dir, f'thresholds_map_fold{fold_idx}.txt'))
        shutil.copyfile(oof_save_path, os.path.join(meta_dir, f'oof_predictions_map_fold{fold_idx}.pt'))
    else:
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, f'best_model_map_fold{fold_idx}.pth'), map_location=DEVICE, weights_only=True))
        _, _, m_targs, m_probs, _, m_att = validate(model, val_loader, DEVICE, label_cols, return_attention=True)
        m_ths = find_optimal_thresholds(m_targs, m_probs, label_cols)
        with open(os.path.join(meta_dir, f'thresholds_map_fold{fold_idx}.txt'), 'w') as f:
            for idx, th in enumerate(m_ths): f.write(f"{label_cols[idx]},{th:.4f}\n")
        torch.save({'probs': m_probs, 'targets': m_targs, 'wsi_ids': valid_wsi_ids, 'thresholds': np.array(m_ths, dtype=np.float32), 'attentions': {sid: att for sid, att in zip(valid_wsi_ids, m_att)}}, os.path.join(meta_dir, f'oof_predictions_map_fold{fold_idx}.pt'))

    # 释放显存，保障 5 折顺序执行时无显存累积泄露
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()

# --- 在 main 函数之前添加这个绘图函数 ---
def plot_history(history, save_path):
    """
    绘制训练过程的静态曲线图 (Robust版)
    """
    epochs = range(1, len(history['loss']) + 1)
    # print(f"DEBUG: Plotting history with {len(history['loss'])} epochs. Loss data: {history['loss'][:5]}...") # Debug log
    
    plt.figure(figsize=(15, 10))
    
    # 1. Loss 曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['loss'], 'b-', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 2. mAP & F1 曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['map'], 'r-', label='Val mAP')
    plt.plot(epochs, history['f1'], 'g-', label='Val Macro F1')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # 3. 学习率曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['lr'], 'y-', label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.grid(True)
    
    # 4. 稀有类别 F1 监控 (带判空保护)
    plt.subplot(2, 2, 4)
    
    # 定义要画的键名和标签
    rare_map = {
        'rare_mucinous': 'Mucinous',
        'rare_micro_invasive': 'Micro-invasive',
        'rare_micropapillary': 'Micropapillary',
        'rare_lymphoma': 'Lymphoma',
        'rare_fea': 'FEA'
    }
    
    has_plot = False
    for key, label in rare_map.items():
        # 只有当 key 存在 且 数据长度等于 epoch 数时才画
        if key in history and len(history[key]) == len(epochs):
            plt.plot(epochs, history[key], label=label)
            has_plot = True
            
    plt.title('Rare Class F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    if has_plot: plt.legend() # 只有画了线才显示图例
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"训练曲线图已保存至: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Multi-Branch CLAM-MB Training")
    parser.add_argument('--csv_path', type=str, default='final_train_list_multilabel.csv')
    parser.add_argument('--backbone', type=str, default='virchow2', 
                        choices=['virchow2', 'conch', 'conch1_5','h_optimus_1', 'gigapath'],
                        help="选择训练的基础模型特征")
    parser.add_argument('--fold', type=int, default=-1, help="指定运行单折 (0-4)，设为 -1 则自动顺序运行全部 5 折")
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f"train_stage3_{args.backbone}_clammb"

    set_seed(SEED)
    df = pd.read_csv(args.csv_path)
    # 强制转为去除首尾空格的字符串后再过滤 'test'
    df = df[df['fold'].astype(str).str.strip().str.lower() != 'test'].copy()
    df['fold'] = df['fold'].astype(int)
    
    label_cols = [c for c in df.columns if c.startswith('label_')]
    n_classes = len(label_cols)
    print(f"🚀 启动训练任务 [Backbone: {args.backbone.upper()}] | 检测到 {n_classes} 个主任务类别")
    
    folds_to_run = range(5) if args.fold == -1 else [args.fold]
    for fold in folds_to_run:
        run_fold(fold, args, df, label_cols, n_classes)
        
    print(f"\n✅ 训练任务完成！权重与元数据已存至 {args.save_dir}")

    # === 5折交叉验证整体结果汇总 (含默认阈值与最优阈值双指标) ===
    if args.fold == -1:
        print("\n" + "="*30 + " 5折交叉验证整体结果汇总 " + "="*30)
        meta_dir = os.path.join(args.save_dir, 'metadata')
        summary_records = []
        all_oof_targets, all_oof_probs, all_oof_thresholds = [], [], []

        for f_idx in range(5):
            oof_path = os.path.join(meta_dir, f'oof_predictions_fold{f_idx}.pt')
            if not os.path.exists(oof_path):
                continue
            data = torch.load(oof_path, map_location='cpu', weights_only=False)
            targs, probs = data['targets'], data['probs']
            ths = data.get('thresholds', np.full(targs.shape[1], 0.5))

            if hasattr(targs, 'cpu'): targs = targs.cpu().numpy()
            if hasattr(probs, 'cpu'): probs = probs.cpu().numpy()
            if hasattr(ths, 'cpu'): ths = ths.cpu().numpy()

            all_oof_targets.append(targs)
            all_oof_probs.append(probs)
            all_oof_thresholds.append(ths)

            # 单折指标计算
            valid_pos = targs.sum(axis=0) > 0
            f_map = (
                average_precision_score(
                    targs[:, valid_pos], probs[:, valid_pos], average="macro"
                )
                if valid_pos.sum() > 0
                else 0.0
            )
            f_map = float(np.nan_to_num(f_map, nan=0.0))
            f_macro_f1_default = f1_score(targs, (probs > 0.5).astype(int), average='macro', zero_division=0)
            f_macro_f1_opt = f1_score(targs, (probs > ths).astype(int), average='macro', zero_division=0)
            
            summary_records.append({
                'Fold': f"Fold {f_idx}",
                'mAP': round(float(f_map), 4),
                'Macro_F1 (th=0.5)': round(float(f_macro_f1_default), 4),
                'Macro_F1 (Opt_Th)': round(float(f_macro_f1_opt), 4)
            })

        if len(summary_records) > 0:
            df_summary = pd.DataFrame(summary_records)
            
            # 全局 OOF 评价
            global_targets = np.vstack(all_oof_targets)
            global_probs = np.vstack(all_oof_probs)
            mean_ths = np.mean(all_oof_thresholds, axis=0)
            global_valid_pos = global_targets.sum(axis=0) > 0
            global_map = (
                average_precision_score(
                    global_targets[:, global_valid_pos], global_probs[:, global_valid_pos], average="macro"
                )
                if global_valid_pos.sum() > 0
                else 0.0
            )
            global_map = float(np.nan_to_num(global_map, nan=0.0))
            global_f1_def = f1_score(global_targets, (global_probs > 0.5).astype(int), average='macro', zero_division=0)
            global_f1_opt = f1_score(global_targets, (global_probs > mean_ths).astype(int), average='macro', zero_division=0)

            # 打印与保存
            print(df_summary.to_string(index=False))
            print("-" * 65)
            print(f"5-Fold Mean ± Std : mAP: {df_summary['mAP'].mean():.4f} ± {df_summary['mAP'].std():.4f} | "
                  f"F1(Opt): {df_summary['Macro_F1 (Opt_Th)'].mean():.4f} ± {df_summary['Macro_F1 (Opt_Th)'].std():.4f}")
            print(f"Global OOF (N={len(global_targets)}): mAP: {global_map:.4f} | F1(Opt): {global_f1_opt:.4f}")

            # 将汇总指标直接追加到 DataFrame 并保存
            df_export = pd.concat([
                df_summary,
                pd.DataFrame([{
                    'Fold': 'Mean ± Std',
                    'mAP': f"{df_summary['mAP'].mean():.4f} ± {df_summary['mAP'].std():.4f}",
                    'Macro_F1 (th=0.5)': f"{df_summary['Macro_F1 (th=0.5)'].mean():.4f} ± {df_summary['Macro_F1 (th=0.5)'].std():.4f}",
                    'Macro_F1 (Opt_Th)': f"{df_summary['Macro_F1 (Opt_Th)'].mean():.4f} ± {df_summary['Macro_F1 (Opt_Th)'].std():.4f}"
                }, {
                    'Fold': f'Global OOF (N={len(global_targets)})',
                    'mAP': round(float(global_map), 4),
                    'Macro_F1 (th=0.5)': round(float(global_f1_def), 4),
                    'Macro_F1 (Opt_Th)': round(float(global_f1_opt), 4)
                }])
            ], ignore_index=True)

            report_csv_path = os.path.join(args.save_dir, 'kfold_summary_report.csv')
            df_export.to_csv(report_csv_path, index=False)
            print(f"✅ 完整汇总报告已自动保存至: {report_csv_path}")

if __name__ == "__main__":
    main()