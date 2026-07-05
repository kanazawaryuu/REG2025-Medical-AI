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
from torch.cuda.amp import autocast, GradScaler # 【新增】混合精度训练

# --- 1. 配置与超参数 ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- 2. ASL Loss (解决长尾多标签的核心) ---
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        x: input logits
        y: targets (multi-label binarized vector)
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

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

class GatedAttentionMIL(nn.Module):
    def __init__(self, n_classes, input_dim=1536, hidden_dim=512, dropout=0.25):
        super(GatedAttentionMIL, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, 256), nn.Sigmoid())
        self.attention_weights = nn.Linear(256, 1)
        
        # --- 多任务头 (升级为 MLP) ---
        self.head_main = nn.Linear(hidden_dim, n_classes) # 主任务保持简单线性即可，避免过拟合
        
        # NST 子任务 (3分类: Grade 1,2,3)
        self.head_nst_tubule = PredictionHead(hidden_dim, 3)
        self.head_nst_nuclear = PredictionHead(hidden_dim, 3)
        self.head_nst_mitoses = PredictionHead(hidden_dim, 3)
        
        # DCIS 子任务
        self.head_dcis_grade = PredictionHead(hidden_dim, 3)    # 3分类 (L,I,H)
        self.head_dcis_necrosis = PredictionHead(hidden_dim, 3) # 3分类 (Absent,Focal,Comedo)
        self.head_dcis_types = PredictionHead(hidden_dim, 3)    # 多标签 (Solid,Cribriform,Micro)

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor(x)
        A = self.attention_weights(self.attention_V(H) * self.attention_U(H))
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)
        
        # 输出所有头
        outputs = {
            'main': self.head_main(M),
            'nst_tubule': self.head_nst_tubule(M),
            'nst_nuclear': self.head_nst_nuclear(M),
            'nst_mitoses': self.head_nst_mitoses(M),
            'dcis_grade': self.head_dcis_grade(M),
            'dcis_necrosis': self.head_dcis_necrosis(M),
            'dcis_types': self.head_dcis_types(M)
        }
        return outputs, A, M

# --- 4. Dataset ---
class BreastDataset(Dataset):
    def __init__(self, csv_file, label_cols, input_dim=1536, training=True):
        self.df = pd.read_csv(csv_file)
        # 移除 feature_path 为空的行
        self.df = self.df[self.df['feature_path'].notna()].reset_index(drop=True)
        self.label_cols = label_cols
        self.input_dim = input_dim
        self.training = training
        
        # --- 【新增】定义子任务列名 ---
        self.nst_cols = ['nst_grade_tubule', 'nst_grade_nuclear', 'nst_grade_mitoses']
        self.dcis_grade_col = 'dcis_grade'
        self.dcis_necrosis_col = 'dcis_necrosis'
        self.dcis_type_cols = ['dcis_type_solid', 'dcis_type_cribriform', 'dcis_type_micropapillary']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_path = row['feature_path']
        
        # 1. 加载特征 (保持原来的增强逻辑)
        try:
            features = np.load(feat_path)
            
            if self.training:
                num_patches = features.shape[0]
                if num_patches > 10: 
                    keep_ratio = np.random.uniform(0.8, 1.0)
                    target_num = min(int(num_patches * keep_ratio), 6000)
                    indices = np.random.choice(num_patches, target_num, replace=False)
                    indices.sort()
                    features = features[indices]
                features = torch.from_numpy(features).float()
                
                # 增强
                noise = torch.randn_like(features) * np.random.uniform(0.0, 0.02)
                features = features + noise
                features = features * np.random.uniform(0.95, 1.05)
            else:
                if features.shape[0] > 6000: features = features[:6000]
                features = torch.from_numpy(features).float()
        except Exception as e:
            # 【修改】捕获异常并打印，避免静默失败
            print(f"⚠️ [Error] 加载特征失败: {feat_path} | Err: {e}")
            features = torch.zeros((1, self.input_dim)).float()

        # 2. 主标签
        labels = row[self.label_cols].values.astype(float)
        labels = torch.from_numpy(labels).float()
        
        # 3. 【新增】子任务标签
        aux_targets = {}
        
        # A. NST 分级 (CSV中是1,2,3 -> 转为0,1,2 用于CE; -1保持-1)
        nst_targets = []
        for col in self.nst_cols:
            val = int(row.get(col, -1))
            target = val - 1 if val > 0 else -1
            nst_targets.append(target)
        aux_targets['nst_grades'] = torch.tensor(nst_targets, dtype=torch.long) # [3]
        
        # B. DCIS 核分级 (1,2,3 -> 0,1,2)
        d_grade = int(row.get(self.dcis_grade_col, -1))
        aux_targets['dcis_grade'] = torch.tensor(d_grade - 1 if d_grade > 0 else -1, dtype=torch.long)
        
        # C. DCIS 坏死 (0,1,2 -> 0,1,2 直接用)
        d_necrosis = int(row.get(self.dcis_necrosis_col, -1))
        aux_targets['dcis_necrosis'] = torch.tensor(d_necrosis, dtype=torch.long)
        
        # D. DCIS 结构类型 (0/1 多标签)
        dcis_types = row[self.dcis_type_cols].values.astype(np.float32)
        aux_targets['dcis_types'] = torch.from_numpy(dcis_types)

        return features, labels, aux_targets

def get_strategic_sampler(dataset):
    """
    【修正版】战略性采样器 (中文日志 + 严格权重控制)
    """
    targets = dataset.df[dataset.label_cols].values
    class_sample_count = targets.sum(axis=0)
    class_sample_count = np.maximum(class_sample_count, 1)
    
    weights_per_class = []
    
    for count in class_sample_count:
        if count >= 100:
            # 大类: 基准权重
            w = 1.0
        elif 50 < count < 100:
            # 中等稀有 (黏液癌): 高权重，让它多出现
            w = 1000.0 / count  # 稍微调高一点倍率
        else: # count <= 50
            # 极度稀有 (化生性癌): 低权重，保持克制
            w = 5.0 
            
        weights_per_class.append(w)
        
    weights_per_class = np.array(weights_per_class)
    
    print("\n--- [采样器策略检查] ---")
    for i, col_name in enumerate(dataset.label_cols):
        # 打印关键类别
        if 'mucinous' in col_name or 'metaplastic' in col_name or 'invasive_nst' in col_name:
            print(f"类别 {col_name:<25} (样本数={int(class_sample_count[i]):>3}): 权重 = {weights_per_class[i]:.2f}")
    
    samples_weight = np.zeros(len(dataset))
    for i in range(len(dataset)):
        label_indices = np.where(targets[i] == 1)[0]
        if len(label_indices) > 0:
            samples_weight[i] = max(weights_per_class[label_indices])
        else:
            samples_weight[i] = 0.5
            
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

# --- 5. 训练与验证 (训练函数) ---
def train_one_epoch(model, loader, criterion_main, optimizer, device, label_cols, scaler=None):
    """
    【核心训练函数】多任务版
    包含：
    1. 主任务 Loss (ASL)
    2. NST 分级 Loss (CE, ignore -1)
    3. DCIS 分级/坏死 Loss (CE, ignore -1)
    4. DCIS 类型 Loss (Masked BCE)
    """
    model.train()
    tracker = {'loss_total': 0, 'loss_main': 0, 'loss_nst': 0, 'loss_dcis': 0}
    
    try:
        idx_dcis = label_cols.index('label_dcis')
    except:
        idx_dcis = -1

    # 辅助 Loss
    criterion_aux_ce = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_aux_bce = nn.BCEWithLogitsLoss(reduction='none')

    for features, labels, aux_targets in tqdm(loader, desc="Training"):
        features, labels = features.to(device), labels.to(device)
        
        nst_grades = aux_targets['nst_grades'].to(device)
        dcis_grade = aux_targets['dcis_grade'].to(device)
        dcis_necrosis = aux_targets['dcis_necrosis'].to(device)
        dcis_types = aux_targets['dcis_types'].to(device)

        optimizer.zero_grad()
        outputs, _, _ = model(features) 
        
        # --- 1. 主任务 Loss ---
        loss_main = criterion_main(outputs['main'], labels)
        
        # --- 2. NST 子任务 Loss (带 NaN 保护) ---
        # 检查本 Batch 是否有 NST 样本 (即 nst_grades 不全为 -1)
        # 我们检查第一列即可 (Tubule)，如果有，通常全都有
        mask_nst = (nst_grades[:, 0] != -1)
        if mask_nst.sum() > 0:
            loss_nst = (
                criterion_aux_ce(outputs['nst_tubule'], nst_grades[:, 0]) +
                criterion_aux_ce(outputs['nst_nuclear'], nst_grades[:, 1]) +
                criterion_aux_ce(outputs['nst_mitoses'], nst_grades[:, 2])
            )
        else:
            loss_nst = torch.tensor(0.0).to(device) # 如果全是 -1，Loss 为 0
        
        # --- 3. DCIS Grade/Necrosis Loss (带 NaN 保护) ---
        mask_dcis_g = (dcis_grade != -1)
        if mask_dcis_g.sum() > 0:
            loss_dcis_g = criterion_aux_ce(outputs['dcis_grade'], dcis_grade)
        else:
            loss_dcis_g = torch.tensor(0.0).to(device)

        mask_dcis_n = (dcis_necrosis != -1)
        if mask_dcis_n.sum() > 0:
            loss_dcis_n = criterion_aux_ce(outputs['dcis_necrosis'], dcis_necrosis)
        else:
            loss_dcis_n = torch.tensor(0.0).to(device)
            
        loss_dcis_gn = loss_dcis_g + loss_dcis_n
        
        # --- 4. DCIS Type Loss (Masked) ---
        loss_dcis_t = torch.tensor(0.0).to(device)
        if idx_dcis != -1:
            mask_dcis_lbl = labels[:, idx_dcis].unsqueeze(1)
            if mask_dcis_lbl.sum() > 0:
                raw_loss = criterion_aux_bce(outputs['dcis_types'], dcis_types)
                loss_dcis_t = (raw_loss * mask_dcis_lbl).sum() / (mask_dcis_lbl.sum() + 1e-6)

        loss_dcis_total = loss_dcis_gn + loss_dcis_t
        loss_total = loss_main + 0.5 * (loss_nst + loss_dcis_total)
        
        # 【修改】混合精度 Backward
        if scaler is not None:
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer) # (Optional) Unscale for gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        # 累积记录
        tracker['loss_total'] += loss_total.item()
        tracker['loss_main'] += loss_main.item()
        tracker['loss_nst'] += loss_nst.item()
        tracker['loss_dcis'] += loss_dcis_total.item()
        
    # 计算平均值
    for k in tracker:
        tracker[k] /= len(loader)
        
    return tracker # 返回字典

# --- 5. 训练与验证 (验证函数) ---
def validate(model, loader, device, label_names):
    model.eval()
    
    # 主任务容器
    all_targets = []
    all_probs = []
    
    # 子任务容器
    sub_task_preds = {'nst_n': [], 'nst_t': [], 'nst_m': [], 'dcis_g': [], 'dcis_n': [], 'dcis_t': []}
    sub_task_targets = {'nst_n': [], 'nst_t': [], 'nst_m': [], 'dcis_g': [], 'dcis_n': [], 'dcis_t': []}
    
    with torch.no_grad():
        for features, labels, aux in loader: # 这里必须接收 3 个参数
            features, labels = features.to(device), labels.to(device)
            
            outputs, _, _ = model(features)
            
            # --- 主任务收集 ---
            logits = outputs['main']
            probs = torch.sigmoid(logits)
            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            # --- 子任务收集 (Softmax取最大作为预测类) ---
            # NST
            sub_task_preds['nst_t'].append(outputs['nst_tubule'].argmax(dim=1).cpu().numpy())
            sub_task_targets['nst_t'].append(aux['nst_grades'][:, 0].cpu().numpy()) # Tubule
            
            sub_task_preds['nst_n'].append(outputs['nst_nuclear'].argmax(dim=1).cpu().numpy())
            sub_task_targets['nst_n'].append(aux['nst_grades'][:, 1].cpu().numpy()) # Nuclear
            
            sub_task_preds['nst_m'].append(outputs['nst_mitoses'].argmax(dim=1).cpu().numpy())
            sub_task_targets['nst_m'].append(aux['nst_grades'][:, 2].cpu().numpy()) # Mitoses
            
            # DCIS
            sub_task_preds['dcis_g'].append(outputs['dcis_grade'].argmax(dim=1).cpu().numpy())
            sub_task_targets['dcis_g'].append(aux['dcis_grade'].cpu().numpy())
            
            sub_task_preds['dcis_n'].append(outputs['dcis_necrosis'].argmax(dim=1).cpu().numpy())
            sub_task_targets['dcis_n'].append(aux['dcis_necrosis'].cpu().numpy())
            
            # DCIS Type (多标签，用 sigmoid > 0.5)
            sub_task_preds['dcis_t'].append((torch.sigmoid(outputs['dcis_types']) > 0.5).int().cpu().numpy())
            sub_task_targets['dcis_t'].append(aux['dcis_types'].cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    
    # --- 计算子任务 Metrics (只在 valid 样本上计算) ---
    sub_metrics = {}
    
    # 辅助函数: 计算 Masked F1
    def calc_masked_f1(preds, targs):
        preds = np.concatenate(preds)
        targs = np.concatenate(targs)
        
        # 1. 多标签任务 (DCIS Type): 直接计算 Macro F1
        if len(targs.shape) > 1: 
            return f1_score(targs, preds, average='macro', zero_division=0)
            
        # 2. 单标签任务 (Grade/Necrosis): 过滤掉 -1
        valid_mask = (targs != -1)
        if valid_mask.sum() == 0:
            return 0.0
            
        p_valid = preds[valid_mask]
        t_valid = targs[valid_mask]
        return f1_score(t_valid, p_valid, average='macro', zero_division=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 主任务 Metrics
        val_map = average_precision_score(all_targets, all_probs, average='macro')
        val_preds = (all_probs > 0.5).astype(int)
        val_f1 = f1_score(all_targets, val_preds, average='macro', zero_division=0)
        
        # --- 【修改】计算所有 6 个子任务的 F1 ---
        # NST
        sub_metrics['nst_tubule'] = calc_masked_f1(sub_task_preds['nst_t'], sub_task_targets['nst_t'])
        sub_metrics['nst_nuclear'] = calc_masked_f1(sub_task_preds['nst_n'], sub_task_targets['nst_n'])
        sub_metrics['nst_mitoses'] = calc_masked_f1(sub_task_preds['nst_m'], sub_task_targets['nst_m'])
        
        # DCIS
        sub_metrics['dcis_grade'] = calc_masked_f1(sub_task_preds['dcis_g'], sub_task_targets['dcis_g'])
        sub_metrics['dcis_necrosis'] = calc_masked_f1(sub_task_preds['dcis_n'], sub_task_targets['dcis_n'])
        sub_metrics['dcis_type'] = calc_masked_f1(sub_task_preds['dcis_t'], sub_task_targets['dcis_t'])

        # 稀有类别监控 (保持不变)
        target_cols = ['label_mucinous', 'label_metaplastic', 'label_micropapillary']
        print("\n--- [稀有类别监控] ---")
        for col in target_cols:
            if col in label_names:
                idx = label_names.index(col)
                cls_f1 = f1_score(all_targets[:, idx], val_preds[:, idx], zero_division=0)
                pred_pos = val_preds[:, idx].sum()
                true_pos = all_targets[:, idx].sum()
                print(f"{col:<25}: F1={cls_f1:.4f} (真={int(true_pos)}, 测={int(pred_pos)})")

    return val_map, val_f1, all_targets, all_probs, sub_metrics

def find_optimal_thresholds(targets, probs, label_names):
    print("\n--- 寻找最佳 F1 阈值 ---")
    best_thresholds = []
    n_classes = targets.shape[1]
    
    for i in range(n_classes):
        best_f1 = 0
        best_th = 0.5
        y_true = targets[:, i]
        y_score = probs[:, i]
        
        # 如果该类别在验证集中全是0 (如化生性癌在某折中可能全是0)
        if y_true.sum() == 0:
            best_thresholds.append(0.5) # 默认
            continue

        for th in np.arange(0.1, 0.95, 0.05):
            y_pred = (y_score > th).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = th
        
        best_thresholds.append(best_th)
        print(f"Class {label_names[i]}: Best Th={best_th:.2f}, F1={best_f1:.4f}")
        
    return best_thresholds

# --- 6. 主程序 (修改版：自动运行 5 折) ---

def run_fold(fold_idx, args, df, label_cols, n_classes):
    """
    封装单折训练逻辑 (自动分类存储文件)
    """
    print(f"\n{'='*20} 开始训练 Fold {fold_idx} {'='*20}")
    
    # --- 1. 定义并创建子文件夹 ---
    base_dir = args.save_dir
    ckpt_dir = os.path.join(base_dir, 'checkpoints')
    log_dir = os.path.join(base_dir, 'logs', f'fold{fold_idx}') # TensorBoard 也是文件夹
    plot_dir = os.path.join(base_dir, 'plots')
    meta_dir = os.path.join(base_dir, 'metadata')
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    
    # --- 2. 划分数据与保存临时CSV ---
    train_df = df[df['fold'] != fold_idx]
    val_df = df[df['fold'] == fold_idx]
    
    # 【修改】临时 CSV 存放到 metadata 文件夹
    train_csv = os.path.join(meta_dir, f'temp_train_fold{fold_idx}.csv')
    val_csv = os.path.join(meta_dir, f'temp_val_fold{fold_idx}.csv')
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    print(f"Fold {fold_idx}: Train size={len(train_df)}, Val size={len(val_df)}")
    if len(val_df) == 0: return

    # --- 3. 初始化 TensorBoard ---
    # 【修改】路径指向 logs 子文件夹
    writer = SummaryWriter(log_dir=log_dir)
    
    history = {
        'loss': [], 'map': [], 'f1': [], 'lr': [],
        'rare_mucinous': [], 'rare_metaplastic': [], 'rare_micropapillary': []
    }

    # Dataset & Loader
    train_dataset = BreastDataset(train_csv, label_cols, input_dim=args.input_dim, training=True)
    val_dataset = BreastDataset(val_csv, label_cols, input_dim=args.input_dim, training=False)
    sampler = get_strategic_sampler(train_dataset)
    
    # 【修改】使用 args.num_workers 和 pin_memory
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # 模型初始化
    model = GatedAttentionMIL(n_classes=n_classes, input_dim=args.input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    
    # 【新增】混合精度 Scaler
    scaler = GradScaler()
    
    best_score = 0
    best_epoch = -1
    
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
        writer.add_scalar('Val/F1_NST_Nuclear', sub_metrics['nst_nuclear'], epoch)
        writer.add_scalar('Val/F1_DCIS_Grade', sub_metrics['dcis_grade'], epoch)

        # 记录所有 6 个子任务到 TensorBoard
        writer.add_scalar('SubTask/NST_Tubule', sub_metrics['nst_tubule'], epoch)
        writer.add_scalar('SubTask/NST_Nuclear', sub_metrics['nst_nuclear'], epoch)
        writer.add_scalar('SubTask/NST_Mitoses', sub_metrics['nst_mitoses'], epoch)
        writer.add_scalar('SubTask/DCIS_Grade', sub_metrics['dcis_grade'], epoch)
        writer.add_scalar('SubTask/DCIS_Necrosis', sub_metrics['dcis_necrosis'], epoch)
        writer.add_scalar('SubTask/DCIS_Type', sub_metrics['dcis_type'], epoch)
        
        # --- 2. 关键任务监控 (打印日志 + History记录) ---
        val_preds = (val_probs > 0.5).astype(int)
        
        # 挑选出的 6 大护法 (代表不同梯队)
        monitor_targets = [
            'label_invasive_nst',       # 【常见】基石
            'label_dcis',               # 【常见】易混淆项
            'label_mucinous',           # 【中等】重点加权项
            'label_invasive_lobular',   # 【中等】特殊形态
            'label_metaplastic',        # 【极稀有】化生性
            'label_micropapillary'      # 【极稀有】微乳头
        ]
        
        # 1. 标题放最前
        print(f"\n{'='*15} [Fold {fold_idx} | Ep {epoch+1}/{args.epochs}] {'='*15}")
        
        # 2. 打印主任务 Loss 和 Score
        current_score = 0.4 * val_map + 0.6 * val_f1
        print(f"Total Loss: {train_loss:.4f} (Main:{loss_dict['loss_main']:.3f} NST:{loss_dict['loss_nst']:.3f} DCIS:{loss_dict['loss_dcis']:.3f})")
        print(f"Metrics   : mAP: {val_map:.4f} | F1: {val_f1:.4f} | Score: {current_score:.4f}")
        
        # 3. 打印关键类别 F1
        val_preds = (val_probs > 0.5).astype(int)
        monitor_targets = ['label_invasive_nst', 'label_dcis', 'label_mucinous', 'label_metaplastic', 'label_micropapillary']
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
                
                # History 记录 (保持不变)
                short_name = col.replace('label_', '')
                if short_name in ['mucinous', 'metaplastic', 'micropapillary']:
                     if f'rare_{short_name}' not in history: history[f'rare_{short_name}'] = []
                     history[f'rare_{short_name}'].append(f1)

        # 4. 打印子任务 (左右分栏)
        print("-" * 65)
        print(f"{'NST Sub-tasks':<30} | {'DCIS Sub-tasks':<30}")
        print("-" * 65)
        print(f"Tubule : {sub_metrics['nst_tubule']:.4f}{' '*14} | Grade   : {sub_metrics['dcis_grade']:.4f}")
        print(f"Nuclear: {sub_metrics['nst_nuclear']:.4f}{' '*14} | Necrosis: {sub_metrics['dcis_necrosis']:.4f}")
        print(f"Mitoses: {sub_metrics['nst_mitoses']:.4f}{' '*14} | Type    : {sub_metrics['dcis_type']:.4f}")
        print("-" * 65)
        
        history['loss'].append(train_loss)
        history['map'].append(val_map)
        history['f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        # 计算当前综合得分
        current_score = 0.4 * val_map + 0.6 * val_f1
        
        # 保存最佳模型
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            save_path = os.path.join(ckpt_dir, f'best_model_fold{fold_idx}.pth')
            torch.save(model.state_dict(), save_path)
            print(f">>> 🏆 最佳模型已保存! (新高分: {best_score:.4f})")
        
        # 【新增】保存每一轮的 latest 模型 (防止中断)
        last_path = os.path.join(ckpt_dir, f'last_model_fold{fold_idx}.pth')
        torch.save(model.state_dict(), last_path)
        
    print(f">>> Fold {fold_idx} 结束. 最佳轮次: {best_epoch+1}, 最高分: {best_score:.4f}")
    
    # 5. 收尾工作
    writer.close()
    
    # 【修改】图片保存到 plots 文件夹
    plot_save_path = os.path.join(plot_dir, f'training_curves_fold{fold_idx}.png')
    plot_history(history, plot_save_path)
    
    # 6. 阈值搜索
    print(f"正在计算 Fold {fold_idx} 的最佳阈值...")
    # 【修改】从 checkpoints 读取模型
    model_path = os.path.join(ckpt_dir, f'best_model_fold{fold_idx}.pth')
    model.load_state_dict(torch.load(model_path))
    
    _, _, val_targets, val_probs, _ = validate(model, val_loader, DEVICE, label_cols)
    best_thresholds = find_optimal_thresholds(val_targets, val_probs, label_cols)
    
    # 【修改】阈值文件保存到 metadata 文件夹
    th_save_path = os.path.join(meta_dir, f'thresholds_fold{fold_idx}.txt')
    with open(th_save_path, 'w') as f:
        for idx, th in enumerate(best_thresholds):
            f.write(f"{label_cols[idx]},{th}\n")
    print(f"阈值已保存至 {th_save_path}")
    
    # 清理临时 CSV
    if os.path.exists(train_csv): os.remove(train_csv)
    if os.path.exists(val_csv): os.remove(val_csv)

# --- 在 main 函数之前添加这个绘图函数 ---
def plot_history(history, save_path):
    """
    绘制训练过程的静态曲线图 (Robust版)
    """
    epochs = range(1, len(history['loss']) + 1)
    print(f"DEBUG: Plotting history with {len(history['loss'])} epochs. Loss data: {history['loss'][:5]}...") # Debug log
    
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
        'rare_metaplastic': 'Metaplastic',
        'rare_micropapillary': 'Micropapillary'
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='final_train_list_multilabel.csv')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--input_dim', type=int, default=1536, help='Feature input dimension')
    # 【修改】默认文件夹名称改为 train_stage3_attention
    parser.add_argument('--save_dir', type=str, default='train_stage3_attention')
    args = parser.parse_args()

    set_seed(SEED)
    # 根目录创建逻辑已移至 run_fold，但这里创建一下也没事
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(args.csv_path)
    df = df[df['fold'] != 'test']
    df['fold'] = df['fold'].astype(int)
    
    label_cols = [c for c in df.columns if c.startswith('label_')]
    n_classes = len(label_cols)
    print(f"检测到 {n_classes} 个类别。准备开始 5 折训练...")
    
    # 循环运行 5 折
    for fold in range(5):
        run_fold(fold, args, df, label_cols, n_classes)
        
    print(f"\n✅ 所有 5 折训练全部完成！结果已保存在 {args.save_dir} 文件夹中。")

if __name__ == "__main__":
    main()