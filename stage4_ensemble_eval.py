import sys
import numpy as np
# 兼容 DGX 端 NumPy 2.x 生成的序列化张量
try:
    import numpy._core
except ImportError:
    import numpy.core as _core
    sys.modules['numpy._core'] = _core
    sys.modules['numpy._core.multiarray'] = _core.multiarray
import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, fbeta_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') # 【新增】WSL多进程防句柄溢出

# --- 1. 设备与环境配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 模型定义 (与 Stage 3 训练严格一致) ---
class PredictionHead(nn.Module):
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
        
        # 辅助多任务预测头
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
        A = self.attention_weights(self.attention_V(H) * self.attention_U(H))
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, H)
        logits_main = (M * self.classifiers.weight).sum(dim=1) + self.classifiers.bias
        logits_main = logits_main.unsqueeze(0)
        
        M_nst = M[self.idx_nst:self.idx_nst+1]
        M_dcis = M[self.idx_dcis:self.idx_dcis+1]
        
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

# --- 3. 动态特征推理 Dataset (安全增强版) ---
class SingleBackboneDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feat_col: str, expected_dim: int):
        self.df = df.reset_index(drop=True)
        self.feat_col = feat_col
        self.expected_dim = expected_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_path = row[self.feat_col]
        
        # 安全提取原始切片 ID (优先使用 Stage 2.2 确立的标准 wsi_id，统一去除 .tiff / .svs 后缀)
        if 'wsi_id' in row and pd.notna(row['wsi_id']):
            wsi_id = Path(str(row['wsi_id'])).stem
        elif 'id' in row and pd.notna(row['id']):
            wsi_id = Path(str(row['id'])).stem
        else:
            wsi_id = f"slide_{idx}"
        
        # 显式路径防崩拦截
        if not os.path.exists(str(feat_path)):
            warnings.warn(f"⚠️ [测试集警报] 样本 {wsi_id} 的特征文件不存在: {feat_path}，将采用保底零张量！")
            return wsi_id, torch.zeros((1, self.expected_dim), dtype=torch.float32)
            
        try:
            features = np.load(feat_path).astype(np.float32).copy()
            features = torch.from_numpy(features).float()

            # 兼容单瓦片一维向量降维情况: [D] -> [1, D]
            if features.ndim == 1:
                if features.shape[0] == self.expected_dim:
                    features = features.unsqueeze(0)
                else:
                    warnings.warn(f"⚠️ [维度异常] 样本 {wsi_id} 一维特征长度 {features.shape[0]} 与预期 {self.expected_dim} 不符，采用保底零张量！")
                    return wsi_id, torch.zeros((1, self.expected_dim), dtype=torch.float32)
            elif features.ndim == 2:
                if features.shape[1] != self.expected_dim:
                    warnings.warn(f"⚠️ [维度不匹配] 样本 {wsi_id} 特征通道 {features.shape[1]} 与模型预期 {self.expected_dim} 不符，采用保底零张量！")
                    return wsi_id, torch.zeros((1, self.expected_dim), dtype=torch.float32)
            else:
                warnings.warn(f"⚠️ [形状异常] 样本 {wsi_id} 特征张量形状为 {features.shape}，采用保底零张量！")
                return wsi_id, torch.zeros((1, self.expected_dim), dtype=torch.float32)

            if features.shape[0] == 0:
                warnings.warn(f"⚠️ [空特征警报] 样本 {wsi_id} 瓦片数量为 0，采用保底零张量！")
                return wsi_id, torch.zeros((1, self.expected_dim), dtype=torch.float32)

        except Exception as e:
            warnings.warn(f"⚠️ [读取异常] 样本 {wsi_id} 读取错误: {e}")
            features = torch.zeros((1, self.expected_dim), dtype=torch.float32)
            
        return wsi_id, features

# --- 4. 骨干模型预测提取器 ---
def _get_available_folds(model_dir, track='score'):
    """检测模型目录中 checkpoints/ 下已就绪的折次列表"""
    if not model_dir or not os.path.isdir(model_dir):
        return []
    ckpt_dir = os.path.join(model_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return []
    suffix = "_map" if track == 'map' else ""
    return [fold for fold in range(5) if os.path.exists(os.path.join(ckpt_dir, f'best_model{suffix}_fold{fold}.pth'))]

def _has_valid_checkpoints(model_dir, require_all_folds=False, track='score'):
    """检测模型目录是否包含可用折权重文件 (支持 5 折完整性校验)"""
    avail = _get_available_folds(model_dir, track=track)
    if require_all_folds:
        return len(avail) == 5
    return len(avail) > 0

def predict_backbone_5fold(model_dir, backbone_name, input_dim, df, label_cols, track='score'):
    """载入指定骨干的 5 折模型，对 df 进行全量推理并返回平均概率"""
    # 【加固】自动兼容短横线与下划线特征列名
    feat_col = f"{backbone_name}_feature_path"
    if feat_col not in df.columns:
        candidates_col = [
            feat_col.replace('_', '-'),
            feat_col.replace('-', '_'),
            feat_col.replace('conch1_5', 'conch_1_5'),
            feat_col.replace('conch1_5', 'conch-1-5'),
            feat_col.replace('conch1_5', 'conch1.5'),
            feat_col.replace('conch_1_5', 'conch1_5'),
        ]
        found_col = next((c for c in candidates_col if c in df.columns), None)
        if found_col:
            feat_col = found_col
        else:
            raise KeyError(f"数据表中缺少特征列: {feat_col} (当前可用特征列: {[c for c in df.columns if 'feature' in c]})")

    dataset = SingleBackboneDataset(df, feat_col, input_dim)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    n_classes = len(label_cols)
    idx_nst = label_cols.index('label_invasive_nst') if 'label_invasive_nst' in label_cols else 0
    idx_dcis = label_cols.index('label_dcis') if 'label_dcis' in label_cols else 1

    # 1. 加载 5 折权重
    models = []
    ckpt_dir = os.path.join(model_dir, 'checkpoints')
    suffix = "_map" if track == 'map' else ""
    for fold in range(5):
        path = os.path.join(ckpt_dir, f'best_model{suffix}_fold{fold}.pth')
        if not os.path.exists(path):
            continue
        model = MultiBranchAttentionMIL(n_classes=n_classes, input_dim=input_dim, idx_nst=idx_nst, idx_dcis=idx_dcis).to(DEVICE)
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        except Exception:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models.append(model)
        
    if not models:
        raise FileNotFoundError(f"❌ 目录 {model_dir} 中未找到任何可用的 5 折模型权重 ({track} 轨)！")
    if len(models) < 5:
        missing_ckpts = [f for f in range(5) if not os.path.exists(os.path.join(ckpt_dir, f'best_model{suffix}_fold{f}.pth'))]
        warnings.warn(f"⚠️ [{backbone_name.upper()}] 仅载入 {len(models)}/5 折权重 (缺失折次: {missing_ckpts})，推断将基于部分折集成！")
    else:
        print(f"[{backbone_name.upper()}] 成功载入完整 5 折集成模型 ({track} 轨)，开始推断...")

    # 2. 读取 5 折最佳阈值求平均 (优先从 .pt 直读浮点数，兼顾 .txt 兜底)
    ths = []
    meta_dir = os.path.join(model_dir, 'metadata')
    for fold in range(5):
        pt_path = os.path.join(meta_dir, f'oof_predictions{suffix}_fold{fold}.pt')
        th_path = os.path.join(meta_dir, f'thresholds{suffix}_fold{fold}.txt')
        
        loaded = False
        if os.path.exists(pt_path):
            try:
                pt_data = torch.load(pt_path, map_location='cpu', weights_only=False)
                if 'thresholds' in pt_data and len(pt_data['thresholds']) == n_classes:
                    ths.append(np.array(pt_data['thresholds'], dtype=np.float32))
                    loaded = True
            except Exception:
                pass
                
        if not loaded and os.path.exists(th_path):
            th_dict = {}
            with open(th_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        th_dict[parts[0].strip()] = float(parts[1].strip())
            if len(th_dict) == n_classes:
                ths.append([th_dict.get(col, 0.5) for col in label_cols])
                
    avg_thresholds = np.mean(ths, axis=0) if ths else np.full(n_classes, 0.5)

    # 3. 逐切片推理 (开启 AMP 加速)
    all_main_probs = []
    all_subtasks = {k: [] for k in ['nst_tubule', 'nst_nuclear', 'nst_mitoses', 'nst_overall', 'dcis_grade', 'dcis_necrosis', 'dcis_types']}
    all_wsi_ids = []

    is_cuda = torch.cuda.is_available() and "cuda" in str(DEVICE)
    device_type = 'cuda' if is_cuda else 'cpu'
    amp_dtype = torch.bfloat16 if (is_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.no_grad():
        for wsi_id, features in tqdm(loader, desc=f"Inference {backbone_name}", leave=False):
            features = features.to(DEVICE)
            all_wsi_ids.append(wsi_id[0] if isinstance(wsi_id, (tuple, list)) else wsi_id)
            
            p_main = torch.zeros(1, n_classes, device=DEVICE)
            p_subs = {k: torch.zeros(1, 3, device=DEVICE) for k in all_subtasks}

            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=is_cuda):
                for model in models:
                    outputs, _, _ = model(features)
                    p_main += torch.sigmoid(outputs['main'].float())
                    for sub_k in ['nst_tubule', 'nst_nuclear', 'nst_mitoses', 'nst_overall', 'dcis_grade', 'dcis_necrosis']:
                        p_subs[sub_k] += torch.softmax(outputs[sub_k].float(), dim=1)
                    p_subs['dcis_types'] += torch.sigmoid(outputs['dcis_types'].float())

            # 求当前样本在 5 折下的均值
            p_main /= len(models)
            all_main_probs.append(p_main.cpu().numpy()[0])
            for k in all_subtasks:
                all_subtasks[k].append((p_subs[k] / len(models)).cpu().numpy()[0])

    del models
    torch.cuda.empty_cache()

    return {
        'wsi_ids': all_wsi_ids,
        'probs_main': np.array(all_main_probs),
        'subtasks': {k: np.array(v) for k, v in all_subtasks.items()},
        'thresholds': np.array(avg_thresholds)
    }

# --- 5. 评测计算辅助函数 ---
def compute_metrics(probs_main, subtasks, targets_main, aux_targets, thresholds, label_cols):
    """计算切片级主任务与子任务评测指标 (与 Stage 3 掩码口径严格对齐，含 AUROC)"""
    valid_pos = targets_main.sum(axis=0) > 0
    # 至少包含 1 例正样本和 1 例负样本的类别方可安全计算 AUROC
    valid_auc = valid_pos & ((1.0 - targets_main).sum(axis=0) > 0)

    if valid_pos.sum() > 0:
        val_map = average_precision_score(targets_main[:, valid_pos], probs_main[:, valid_pos], average='macro')
        val_map = float(np.nan_to_num(val_map, nan=0.0))
        macro_f1_default = f1_score(targets_main[:, valid_pos], (probs_main[:, valid_pos] > 0.5).astype(int), average='macro', zero_division=0)
        macro_f1_opt = f1_score(targets_main[:, valid_pos], (probs_main[:, valid_pos] > thresholds[valid_pos]).astype(int), average='macro', zero_division=0)
    else:
        val_map, macro_f1_default, macro_f1_opt = 0.0, 0.0, 0.0

    val_auc = 0.0
    if valid_auc.sum() > 0:
        try:
            raw_auc = roc_auc_score(targets_main[:, valid_auc], probs_main[:, valid_auc], average='macro')
            val_auc = float(np.nan_to_num(raw_auc, nan=0.0))
        except Exception:
            # 极端边界兜底：逐类安全容错计算 (杜绝单类无分离度导致的全局崩溃)
            auc_list = []
            for c_idx in np.where(valid_auc)[0]:
                try:
                    s = roc_auc_score(targets_main[:, c_idx], probs_main[:, c_idx])
                    if not np.isnan(s):
                        auc_list.append(s)
                except Exception:
                    continue
            val_auc = float(np.mean(auc_list)) if auc_list else 0.0

    def _sub_f1(preds, targs):
        mask = (targs != -1)
        if mask.sum() == 0: return 0.0
        return f1_score(targs[mask], preds[mask], average='macro', zero_division=0)

    nst_tubule_f1 = _sub_f1(subtasks['nst_tubule'].argmax(axis=1), aux_targets.get('nst_tubule', np.full(len(probs_main), -1)))
    nst_nuclear_f1 = _sub_f1(subtasks['nst_nuclear'].argmax(axis=1), aux_targets.get('nst_nuclear', np.full(len(probs_main), -1)))
    nst_mitoses_f1 = _sub_f1(subtasks['nst_mitoses'].argmax(axis=1), aux_targets.get('nst_mitoses', np.full(len(probs_main), -1)))
    nst_overall_f1 = _sub_f1(subtasks['nst_overall'].argmax(axis=1), aux_targets.get('nst_overall', np.full(len(probs_main), -1)))
    dcis_grade_f1 = _sub_f1(subtasks['dcis_grade'].argmax(axis=1), aux_targets.get('dcis_grade', np.full(len(probs_main), -1)))
    dcis_necrosis_f1 = _sub_f1(subtasks['dcis_necrosis'].argmax(axis=1), aux_targets.get('dcis_necrosis', np.full(len(probs_main), -1)))

    idx_dcis = label_cols.index('label_dcis') if 'label_dcis' in label_cols else -1
    dcis_mask = (aux_targets['dcis_grade'] != -1) if idx_dcis == -1 else ((aux_targets['dcis_grade'] != -1) | (targets_main[:, idx_dcis] >= 0.5))

    if dcis_mask.sum() > 0:
        dcis_type_f1 = f1_score(
            aux_targets['dcis_types'][dcis_mask], 
            (subtasks['dcis_types'][dcis_mask] > 0.5).astype(int), 
            average='macro', zero_division=0
        )
    else:
        dcis_type_f1 = 0.0

    # 计算 22 个类别的独立 F1，供细粒度报表透视
    per_cls_f1 = {}
    preds_opt = (probs_main > thresholds).astype(int)
    for c_idx, c_name in enumerate(label_cols):
        per_cls_f1[c_name] = round(float(f1_score(targets_main[:, c_idx], preds_opt[:, c_idx], zero_division=0)), 4)

    return {
        'mAP': round(float(val_map), 4),
        'AUROC': round(float(val_auc), 4),
        'F1@0.5': round(float(macro_f1_default), 4),
        'F1@Opt': round(float(macro_f1_opt), 4),
        'NST_Tubule_F1': round(float(nst_tubule_f1), 4),
        'NST_Nuclear_F1': round(float(nst_nuclear_f1), 4),
        'NST_Mitoses_F1': round(float(nst_mitoses_f1), 4),
        'NST_Overall_F1': round(float(nst_overall_f1), 4),
        'DCIS_Grade_F1': round(float(dcis_grade_f1), 4),
        'DCIS_Necrosis_F1': round(float(dcis_necrosis_f1), 4),
        'DCIS_Type_F1': round(float(dcis_type_f1), 4),
        '_per_class_f1': per_cls_f1
    }

# --- 6. 临床病理报告文本生成 (对接条件阈值字典) ---
def generate_report_text(main_prob, sub_probs, thresholds, label_cols, cascade_params=None):
    # 彻底解耦人工参数：直接获取 OOF 标定的条件阈值字典与拦截界限
    cond_thresholds = cascade_params.get('cond_thresholds', {}) if cascade_params else {}
    beta = cascade_params.get('beta', 0.25) if cascade_params else 0.25

    text_map = {
        'label_invasive_nst': "Invasive carcinoma of no special type",
        'label_invasive_lobular': "Invasive lobular carcinoma",
        'label_mucinous': "Mucinous carcinoma",
        'label_micropapillary': "Invasive micropapillary carcinoma",
        'label_micro_invasive': "Micro-invasive carcinoma",
        'label_dcis': "Ductal carcinoma in situ",
        'label_lcis': "Lobular carcinoma in situ",
        'label_fibroepithelial': "Fibroepithelial tumor",
        'label_fibroadenoma': "Fibroadenoma",
        'label_phyllodes': "Phyllodes tumor",
        'label_papillary_neoplasm': "Papillary neoplasm",
        'label_papilloma': "Intraductal papilloma",
        'label_adh': "Atypical ductal hyperplasia",
        'label_fea': "Flat epithelial atypia",
        'label_udh': "Usual ductal hyperplasia",
        'label_columnar': "Columnar cell lesion",
        'label_sclerosing_adenosis': "Sclerosing adenosis",
        'label_lymphoma': "Malignant lymphoma",
        'label_microcalcification': "Microcalcification",
        'label_fibrocystic': "Fibrocystic change",
        'label_apocrine_metaplasia': "Apocrine metaplasia",
        'label_no_tumor': "No evidence of tumor"
    }

    tier1_invasive = ['label_invasive_nst', 'label_invasive_lobular', 'label_mucinous', 'label_micropapillary', 'label_micro_invasive', 'label_lymphoma']
    tier2_insitu = ['label_dcis', 'label_lcis']
    tier3_neoplasm = ['label_fibroadenoma', 'label_phyllodes', 'label_fibroepithelial', 'label_papillary_neoplasm', 'label_papilloma']
    tier4_atypia = ['label_adh', 'label_fea']
    tier5_benign = ['label_udh', 'label_columnar', 'label_sclerosing_adenosis', 'label_fibrocystic', 'label_apocrine_metaplasia']

    # 1. 筛选独立过线阳性病灶并追踪最大阳性候选
    active_positive = []
    max_pos_prob = -1.0
    max_pos_label = None

    for i, cls in enumerate(label_cols):
        if cls in ['label_no_tumor', 'label_microcalcification']:
            continue
        p = float(main_prob[i])
        th = float(thresholds[i] if isinstance(thresholds, (list, np.ndarray)) else thresholds.get(cls, 0.5))
        
        if p > max_pos_prob:
            max_pos_prob = p
            max_pos_label = cls

        if p > th:
            active_positive.append((cls, p - th))

    # 2. 仲裁第一主诊断
    primary_label = None
    idx_no_tumor = label_cols.index('label_no_tumor') if 'label_no_tumor' in label_cols else -1
    prob_no_tumor = float(main_prob[idx_no_tumor]) if idx_no_tumor != -1 else 0.0

    if active_positive:
        active_names = [x[0] for x in active_positive]
        cands = active_positive
        if 'label_fibroadenoma' in active_names or 'label_phyllodes' in active_names:
            cands = [x for x in cands if x[0] != 'label_fibroepithelial']
        if 'label_papillary_neoplasm' in active_names:
            cands = [x for x in cands if x[0] != 'label_papilloma']

        t1_cands = [x for x in cands if x[0] in tier1_invasive]
        t2_cands = [x for x in cands if x[0] in tier2_insitu]
        t3_cands = [x for x in cands if x[0] in tier3_neoplasm]
        t4_cands = [x for x in cands if x[0] in tier4_atypia]
        t5_cands = [x for x in cands if x[0] in tier5_benign]

        if t1_cands:
            primary_label = max(t1_cands, key=lambda x: x[1])[0]
        elif t2_cands:
            primary_label = max(t2_cands, key=lambda x: x[1])[0]
        elif t3_cands:
            primary_label = max(t3_cands, key=lambda x: x[1])[0]
        elif t4_cands:
            max_t4_prob = float(main_prob[label_cols.index(max(t4_cands, key=lambda x: x[1])[0])])
            primary_label = 'label_no_tumor' if (prob_no_tumor >= 0.50 and prob_no_tumor > max_t4_prob) else max(t4_cands, key=lambda x: x[1])[0]
        elif t5_cands:
            # 无任何真性肿瘤时，仅在阴性概率压过良性候选或显著过线时，判定为未见肿瘤
            best_t5_name = max(t5_cands, key=lambda x: x[1])[0]
            max_t5_prob = float(main_prob[label_cols.index(best_t5_name)])
            th_no_tumor = float(thresholds[idx_no_tumor] if isinstance(thresholds, (list, np.ndarray)) else thresholds.get('label_no_tumor', 0.5))
            if (prob_no_tumor > max_t5_prob) or (prob_no_tumor >= th_no_tumor and prob_no_tumor >= 0.50):
                primary_label = 'label_no_tumor'
            else:
                primary_label = best_t5_name

    # 【OOF 标定边界 beta 拦截保护】：未过线样本杜绝盲目判阴
    if primary_label is None:
        if prob_no_tumor > 0.50 and prob_no_tumor > max_pos_prob:
            primary_label = 'label_no_tumor'
        elif max_pos_prob >= beta and max_pos_label is not None:
            primary_label = max_pos_label
        elif prob_no_tumor >= 0.40:
            primary_label = 'label_no_tumor'
        else:
            primary_label = 'label_invasive_nst'

    header = "Breast, sono guided core biopsy;"
    # 【漏洞 1 修复】：纯阴性标本严格输出单句，杜绝生成带有 "1. No evidence..." 的畸形格式
    if primary_label == 'label_no_tumor':
        return f"{header}\n  No evidence of tumor"

    # 3. 收集伴随特征
    secondary_labels = []
    for i, col in enumerate(label_cols):
        if col == primary_label or col == 'label_no_tumor': 
            continue
        p = float(main_prob[i])
        th = float(thresholds[i] if isinstance(thresholds, (list, np.ndarray)) else thresholds.get(col, 0.5))

        # 【数据驱动查表】
        if col == 'label_microcalcification':
            th = cond_thresholds.get('label_microcalcification', th)
        elif primary_label in tier1_invasive and col in ['label_dcis', 'label_lcis']:
            th = cond_thresholds.get(col, th)

        if p > th:
            secondary_labels.append(col)

    # 病理层级互斥
    if primary_label == 'label_micro_invasive' and 'label_dcis' not in secondary_labels:
        secondary_labels.insert(0, 'label_dcis')
    # 【修复】：一旦第一主诊断为浸润癌，伴随列表中严禁再出现任何浸润癌，杜绝双重侵润癌错误
    if primary_label in tier1_invasive:
        secondary_labels = [c for c in secondary_labels if c not in tier1_invasive]
    if any(k in secondary_labels or primary_label == k for k in ['label_fibroadenoma', 'label_phyllodes']):
        secondary_labels = [c for c in secondary_labels if c != 'label_fibroepithelial']

    # 【修复点 1】：重构乳头状病变抑制逻辑 (纯列表推导式安全过滤，零崩溃)
    if primary_label in ['label_papillary_neoplasm', 'label_papilloma']:
        secondary_labels = [c for c in secondary_labels if c not in ['label_papilloma', 'label_papillary_neoplasm']]
    elif 'label_papilloma' in secondary_labels:
        # 只要存在特称 papilloma，安全剥离上位泛称，杜绝 ValueError
        secondary_labels = [c for c in secondary_labels if c != 'label_papillary_neoplasm']

    # 【新增病理谱系吸收规则】
    # 1) 原位癌 DCIS 直接吸收下位导管增生(ADH/UDH)、硬化性腺病与泛称乳头状病变，防 "\n  2" 粘连
    if primary_label == 'label_dcis':
        secondary_labels = [c for c in secondary_labels if c in ['label_microcalcification', 'label_fibroadenoma', 'label_phyllodes', 'label_lcis']]
    # 2) 泛称乳头状肿瘤(非特称乳头状瘤)不单独拆行罗列伴随 UDH/ADH，防 "papillary neoplasm\n  2" 粘连
    elif primary_label == 'label_papillary_neoplasm' and 'label_papilloma' not in [x[0] for x in active_positive]:
        secondary_labels = [c for c in secondary_labels if c not in ['label_udh', 'label_adh']]

    # 4. 构建主病灶文本与复合修饰处理
    findings = []
    active_names = [x[0] for x in active_positive]
    
    # 【调整 1】仅当明确检出特称 papilloma 时才合并 UDH，保留泛称 Papillary neoplasm 原貌
    if (primary_label == 'label_papilloma' or 'label_papilloma' in active_names) and 'label_udh' in secondary_labels:
        secondary_labels = [c for c in secondary_labels if c != 'label_udh']
        primary_text = "Intraductal papilloma with usual ductal hyperplasia"
    elif primary_label == 'label_fibroadenoma':
        primary_text = "Fibroepithelial tumor, favor fibroadenoma"
    elif primary_label == 'label_phyllodes':
        primary_text = "Fibroepithelial tumor, favor phyllodes tumor"
    else:
        primary_text = text_map.get(primary_label, "Lesion")

    if primary_label == 'label_invasive_nst':
        t_score = int(sub_probs['nst_tubule'].argmax()) + 1
        n_score = int(sub_probs['nst_nuclear'].argmax()) + 1
        m_score = int(sub_probs['nst_mitoses'].argmax()) + 1
        tot = t_score + n_score + m_score
        grade_roman = "I" if tot <= 5 else ("II" if tot <= 7 else "III")
        primary_text = f"{primary_text}, grade {grade_roman} (Tubule formation: {t_score}, Nuclear grade: {n_score}, Mitoses: {m_score})"

    elif primary_label == 'label_dcis':
        g_map = ["Low", "Intermediate", "High"]
        g_text = g_map[int(sub_probs['dcis_grade'].argmax())]
        
        p_nec = sub_probs['dcis_necrosis']
        if (float(p_nec[1]) + float(p_nec[2])) > 0.40 and float(p_nec[0]) <= 0.60:
            n_text = "Present (Comedo-type)" if float(p_nec[2]) >= float(p_nec[1]) else "Present (Focal)"
        else:
            n_text = "Absent"
            
        # 【调整 2】支持 DCIS 双亚型并列输出 (适配三模平滑后的次峰概率 0.32)
        p_t = sub_probs['dcis_types']
        t_map = ["Solid", "Cribriform", "Micropapillary"]
        if float(p_t[0]) >= 0.32 and float(p_t[1]) >= 0.32:
            t_text = "Cribriform and solid"
        elif float(p_t[0]) >= 0.32 and float(p_t[2]) >= 0.32:
            t_text = "Solid and micropapillary"
        elif float(p_t[1]) >= 0.32 and float(p_t[2]) >= 0.32:
            t_text = "Cribriform and micropapillary"
        else:
            t_text = t_map[int(p_t.argmax())]
        
        # 【调整 3】统一固定 2 空格缩进，彻底消除 SciSpacy 换行空格错位扣分
        sub_indent = "  "
        primary_text = f"{primary_text}\n{sub_indent}- Type: {t_text}\n{sub_indent}- Nuclear grade: {g_text}\n{sub_indent}- Necrosis: {n_text}"

    findings.append(primary_text)

    for col in secondary_labels:
        findings.append(text_map.get(col, col))

    # 5. 格式组装：单病灶绝不添加 "1."，多病灶严格标号
    if len(findings) == 0:
        report = f"{header}\n  No evidence of tumor"
    elif len(findings) == 1:
        report = f"{header}\n  {findings[0]}"
    else:
        lines = [header]
        for idx, item in enumerate(findings):
            lines.append(f"  {idx + 1}. {item}")
        report = "\n".join(lines)

    return report    
def load_all_oof_predictions(model_dirs_dict, strict_5fold=True, track='score'):
    """
    【加固版】严格校验 5 折完整性，按 wsi_id 主键对齐多模型 OOF 预测。
    若某模型不足 5 折，显式报警并拦截剔除，严防验证集样本量被不完整模型从 1500+ 拉垮缩水。
    """
    suffix = "_map" if track == 'map' else ""
    raw_oof = {}
    raw_ordered_data = {}
    all_slide_sets = []
    incomplete_models = {}

    for name, m_dir in model_dirs_dict.items():
        meta_dir = os.path.join(m_dir, 'metadata')
        model_dict = {}
        ordered_p, ordered_t, ordered_ids = [], [], []
        found_folds = []
        for fold in range(5):
            pt_path = os.path.join(meta_dir, f'oof_predictions{suffix}_fold{fold}.pt')
            if not os.path.exists(pt_path):
                continue
            try:
                data = torch.load(pt_path, map_location='cpu', weights_only=False)
            except Exception as e:
                warnings.warn(f"⚠️ 读取 OOF 文件失败 {pt_path}: {e}")
                continue
                
            t = data['targets'].cpu().numpy() if hasattr(data['targets'], 'cpu') else data['targets']
            p = data['probs'].cpu().numpy() if hasattr(data['probs'], 'cpu') else data['probs']
            wsi_ids = data.get('wsi_ids', [f"f{fold}_{i}" for i in range(len(p))])
            
            for sid, prob_vec, targ_vec in zip(wsi_ids, p, t):
                clean_sid = Path(str(sid)).stem
                model_dict[clean_sid] = (prob_vec, targ_vec)
                ordered_p.append(prob_vec)
                ordered_t.append(targ_vec)
                ordered_ids.append(clean_sid)
            found_folds.append(fold)
                
        if not model_dict:
            continue

        # 显式校验 5 折完整性
        if len(found_folds) < 5:
            missing_folds = [f for f in range(5) if f not in found_folds]
            msg = (f"⚠️ [OOF 完整性警报] 模型 '{name}' 仅包含 {len(found_folds)}/5 折 OOF 预测 "
                   f"(已就绪: {found_folds}, 缺失: {missing_folds}, 样本量: {len(model_dict)} 例)！")
            if strict_5fold:
                warnings.warn(f"{msg}\n   🔒 为防止公共验证集样本量被严重拉垮缩水，已将 '{name}' 剔除出 OOF 权重搜索与标定！")
                incomplete_models[name] = {
                    'found_folds': found_folds,
                    'missing_folds': missing_folds,
                    'count': len(model_dict),
                    'dict': model_dict,
                    'ordered': {
                        'probs': np.array(ordered_p),
                        'targets': np.array(ordered_t),
                        'wsi_ids': ordered_ids
                    }
                }
                continue
            else:
                warnings.warn(f"{msg}\n   ⚠️ 当前未开启严格拦截，将继续纳入对齐 (注意验证样本量可能大幅缩水)！")

        raw_oof[name] = model_dict
        all_slide_sets.append(set(model_dict.keys()))
        raw_ordered_data[name] = {
            'probs': np.array(ordered_p),
            'targets': np.array(ordered_t),
            'wsi_ids': ordered_ids
        }

    # 【降级回退保护】若所有模型均不足 5 折，解除严格拦截以保证流程不中断
    if not raw_oof and incomplete_models:
        warnings.warn(
            f"⚠️ [OOF 降级回退] 所有模型 ({list(incomplete_models.keys())}) 均不足 5 折！\n"
            f"   为保证流程继续运行，启动降级模式使用现有不完整折次进行对齐 (注意验证样本量受限)。"
        )
        for name, info in incomplete_models.items():
            raw_oof[name] = info['dict']
            all_slide_sets.append(set(info['dict'].keys()))
            raw_ordered_data[name] = info['ordered']

    if not raw_oof:
        return {}

    # 取所有模型共有的切片交集并强制排序，确保行索引 100% 绝对一致
    common_ids = sorted(list(set.intersection(*all_slide_sets))) if all_slide_sets else []
    
    # 【加固】若交集不为空，优先使用主键绝对对齐
    if len(common_ids) > 0:
        print(f"🔒 [OOF 对齐保障] 成功对齐 {len(raw_oof)} 个模型的公共验证切片: {len(common_ids)} 例 (零错位)")
        aligned_oof = {}
        for name in raw_oof:
            probs = np.array([raw_oof[name][sid][0] for sid in common_ids])
            targets = np.array([raw_oof[name][sid][1] for sid in common_ids])
            aligned_oof[name] = {'probs': probs, 'targets': targets, 'wsi_ids': common_ids}
        return aligned_oof

    # 【回退保护】若各模型主键格式差异导致交集为空，安全回退到行号顺序对齐
    warnings.warn("⚠️ [OOF 对齐警报] 模型间切片主键格式不一致导致交集为 0！启动折次与行号顺序对齐回退保护...")
    min_len = min(len(raw_ordered_data[n]['probs']) for n in raw_ordered_data)
    if min_len == 0:
        return {}
    aligned_oof = {}
    for name in raw_ordered_data:
        aligned_oof[name] = {
            'probs': raw_ordered_data[name]['probs'][:min_len],
            'targets': raw_ordered_data[name]['targets'][:min_len],
            'wsi_ids': [f"oof_sample_{k}" for k in range(min_len)]
        }
    print(f"🛡️ [OOF 回退保护已激活] 成功按序列行号对齐 {len(aligned_oof)} 个模型: {min_len} 例")
    return aligned_oof

def calibrate_thresholds(targets, probs, beta=1.0):
    """在连续概率上搜索 22 维各类别的最佳 F_beta 截断阈值"""
    if targets is None or len(targets) == 0 or targets.ndim < 2:
        return np.full(probs.shape[1] if probs is not None and probs.ndim > 1 else 22, 0.5)
    thresholds = []
    n_classes = targets.shape[1]
    targets_int = targets.astype(int)
    
    candidate_ths = np.round(np.linspace(0.02, 0.94, 47), 2)

    for c in range(n_classes):
        y_true = targets_int[:, c]
        y_score = probs[:, c]
        if y_true.sum() == 0:
            thresholds.append(0.5)
            continue
            
        best_th, best_score = 0.5, 0.0
        for th in candidate_ths:
            preds = (y_score > th).astype(int)
            # 根据传入的 beta 自动切换 (主干用 F1，伴随病变用 F2)
            if beta == 1.0:
                score = f1_score(y_true, preds, zero_division=0)
            else:
                score = fbeta_score(y_true, preds, beta=beta, zero_division=0)
                
            if score > best_score:
                best_score, best_th = score, th
                
        if best_score == 0.0:
            best_th = 0.5
            
        thresholds.append(round(float(best_th), 4))
    return np.array(thresholds)

def calibrate_conditional_thresholds(targets, probs, base_thresholds, label_cols):

    if targets is None or len(targets) == 0:
        return {'cond_thresholds': {}, 'beta': 0.25}

    pos_targets = targets.astype(int)
    valid_pos = (pos_targets.sum(axis=0) > 0)
    
    tier1_invasive = ['label_invasive_nst', 'label_invasive_lobular', 'label_mucinous', 'label_micropapillary', 'label_micro_invasive', 'label_lymphoma']
    t1_indices = [label_cols.index(c) for c in tier1_invasive if c in label_cols]
    
    idx_no_tumor = label_cols.index('label_no_tumor') if 'label_no_tumor' in label_cols else -1
    idx_dcis = label_cols.index('label_dcis') if 'label_dcis' in label_cols else -1
    idx_lcis = label_cols.index('label_lcis') if 'label_lcis' in label_cols else -1
    idx_calc = label_cols.index('label_microcalcification') if 'label_microcalcification' in label_cols else -1

    # =========================================================================
    # 步骤 1: 伴随病变温和先验标定 (修复门槛倒挂问题)
    # =========================================================================
    cond_thresholds = {}
    
    alpha_companion = 0.88   # 伴随病变折减系数 (降门槛)
    min_safe_floor  = 0.20   # 真正的防决堤安全下限 (杜绝低于 0.20 的噪声误报)

    base_dcis = float(base_thresholds[idx_dcis]) if idx_dcis != -1 else 0.5
    base_lcis = float(base_thresholds[idx_lcis]) if idx_lcis != -1 else 0.5
    base_calc = float(base_thresholds[idx_calc]) if idx_calc != -1 else 0.5

    # 伴随阈值 = 在基础阈值上打折，但设置 0.20 保底线，且严格不能高于基础阈值自身
    if idx_dcis != -1:
        cond_thresholds['label_dcis'] = max(round(base_dcis * alpha_companion, 2), min_safe_floor)
    if idx_lcis != -1:
        cond_thresholds['label_lcis'] = max(round(base_lcis * alpha_companion, 2), min_safe_floor)
    if idx_calc != -1:
        cond_thresholds['label_microcalcification'] = max(round(base_calc * alpha_companion, 2), min_safe_floor)
    
    # =========================================================================
    # 步骤 2: 判阴拦截下界 beta 的全自动一维求解 (0.10 ~ 0.45 线性搜索最优 F1)
    # =========================================================================
    candidate_betas = np.round(np.linspace(0.10, 0.45, 36), 2)
    best_beta = 0.25
    best_f1 = -1.0

    # 预先构建基础判定矩阵与条件阈值向量
    th_base = base_thresholds.copy()
    
    for beta_cand in candidate_betas:
        preds = np.zeros_like(probs, dtype=int)
        for s_idx in range(len(probs)):
            p_vec = probs[s_idx]
            th_vec = th_base.copy()
            
            # 若切片被浸润癌激活，激活条件阈值
            has_inv = any(p_vec[i] > th_vec[i] for i in t1_indices)
            has_primary = any(p_vec[i] > th_vec[i] for i in range(len(p_vec)) if i not in [idx_no_tumor, idx_calc])
            
            if has_inv:
                if idx_dcis != -1: th_vec[idx_dcis] = cond_thresholds['label_dcis']
                if idx_lcis != -1: th_vec[idx_lcis] = cond_thresholds['label_lcis']
            if has_primary and idx_calc != -1:
                th_vec[idx_calc] = cond_thresholds['label_microcalcification']
                
            over_th = (p_vec > th_vec)
            
            # 主病灶检验
            primary_over_th = over_th.copy()
            if idx_no_tumor != -1: primary_over_th[idx_no_tumor] = False
            if idx_calc != -1: primary_over_th[idx_calc] = False
            
            # 判阴保护：无主病灶过线时，检验最大阳性候选是否越过 beta
            if not primary_over_th.any():
                p_eval = p_vec.copy()
                if idx_no_tumor != -1: p_eval[idx_no_tumor] = -1.0
                if idx_calc != -1: p_eval[idx_calc] = -1.0
                max_pos_idx = int(np.argmax(p_eval))
                
                if p_eval[max_pos_idx] >= beta_cand:
                    over_th[max_pos_idx] = True
                elif idx_no_tumor != -1:
                    over_th[idx_no_tumor] = True
                    
            preds[s_idx] = over_th.astype(int)
            
        score = f1_score(pos_targets[:, valid_pos], preds[:, valid_pos], average='macro', zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_beta = float(beta_cand)

    print(f"🔒 [OOF 条件标定完成] 伴随DCIS={cond_thresholds.get('label_dcis', 0.5):.2f}, "
          f"伴随钙化={cond_thresholds.get('label_microcalcification', 0.5):.2f}, "
          f"拦截下界 beta={best_beta:.2f} (OOF Macro F1: {best_f1:.4f})")

    return {'cond_thresholds': cond_thresholds, 'beta': best_beta}

def optimize_weights_and_thresholds(oof_data, subset_names, track='score'):
    """
    【加固版】根据轨道动态切换目标函数的整数格点搜索 (步长 5%)
    """
    for n in subset_names:
        if n not in oof_data or 'probs' not in oof_data[n] or len(oof_data[n]['probs']) == 0:
            return None, None
            
    # 提取共有安全最小样本行数，杜绝因单切片微弱行数差异导致的矩阵广播崩溃
    min_rows = min(len(oof_data[n]['probs']) for n in subset_names)
    min_rows = min(min_rows, len(oof_data[subset_names[0]]['targets']))
    if min_rows == 0:
        return None, None

    targets = oof_data[subset_names[0]]['targets'][:min_rows].astype(int)
    if targets is None or len(targets) == 0 or targets.ndim < 2:
        return None, None
    valid_pos = targets.sum(axis=0) > 0
    candidate_ths = np.round(np.linspace(0.02, 0.94, 47), 2)
    
    # 提前构造纯布尔目标张量与真实正样本计数，彻底杜绝 NumPy 1.25+/2.0+ int & bool 跨类型位运算崩溃 (DTypePromotionError)
    targets_bool_3d = targets.astype(bool)[:, :, None]
    not_targets_bool_3d = ~targets_bool_3d
    act_p = targets.sum(axis=0)[:, None]

    best_score = -1.0
    best_weights = [round(1.0 / len(subset_names), 3)] * len(subset_names)

    # 向量化评测函数：根据轨道动态切换优化目标
    def _eval_combo_score(ens_p):
        if valid_pos.sum() == 0:
            return 0.0
        cur_map = average_precision_score(targets[:, valid_pos], ens_p[:, valid_pos], average='macro')
        cur_map = float(np.nan_to_num(cur_map, nan=0.0))
        
        # 若为纯 map 轨道，100% 以 mAP 为唯一指标寻优
        if track == 'map':
            return cur_map
        
        # 否则按官方 7:3 综合分寻优
        preds_all = ens_p[:, :, None] > candidate_ths[None, None, :]
        tp = (targets_bool_3d & preds_all).sum(axis=0)
        fp = (not_targets_bool_3d & preds_all).sum(axis=0)
        fn = act_p - tp
        denom = 2 * tp + fp + fn
        f1s = np.where(denom > 0, 2 * tp / denom, 0.0)
        cur_f1_opt = float(f1s.max(axis=1)[valid_pos].mean())
        
        return 0.7 * cur_map + 0.3 * cur_f1_opt

    # 1. 两两配对搜索 (0% ~ 100% 步长 5%，共 21 次遍历)
    if len(subset_names) == 2:
        m1, m2 = subset_names
        p1, p2 = oof_data[m1]['probs'][:min_rows], oof_data[m2]['probs'][:min_rows]
        for i in range(21):
            w1 = i / 20.0
            w2 = round(1.0 - w1, 4)
            ens_p = w1 * p1 + w2 * p2
            cur_score = _eval_combo_score(ens_p)
            if cur_score > best_score:
                best_score = cur_score
                best_weights = [round(w1, 3), round(w2, 3)]
                
    # 2. 三模型大一统搜索 (纯整数离散格点，共 231 次遍历，耗时 ~2 秒)
    elif len(subset_names) == 3:
        m1, m2, m3 = subset_names
        p1, p2, p3 = oof_data[m1]['probs'][:min_rows], oof_data[m2]['probs'][:min_rows], oof_data[m3]['probs'][:min_rows]
        for i in range(21):
            for j in range(21 - i):
                k = 20 - i - j
                w1, w2, w3 = i / 20.0, j / 20.0, k / 20.0
                ens_p = w1 * p1 + w2 * p2 + w3 * p3
                cur_score = _eval_combo_score(ens_p)
                if cur_score > best_score:
                    best_score = cur_score
                    best_weights = [round(w1, 3), round(w2, 3), round(w3, 3)]

    # 显式归一化权重并标定最优决策阈值
    sum_w = sum(best_weights)
    norm_best_weights = [round(w / sum_w, 4) for w in best_weights]
    # 消除浮点累加余量，确保和精确为 1.0
    norm_best_weights[-1] = round(1.0 - sum(norm_best_weights[:-1]), 4)
    fused_p = sum(w * oof_data[n]['probs'][:min_rows] for n, w in zip(subset_names, norm_best_weights))
    best_ths = calibrate_thresholds(targets, fused_p)
    return norm_best_weights, best_ths
    
# --- 7. 主程序入口 ---
def main():
    parser = argparse.ArgumentParser(description="Stage 4: Multi-Model Ensemble and Comprehensive Evaluation")
    parser.add_argument('--csv_path', type=str, default='final_train_list_multilabel.csv')
    parser.add_argument('--virchow2_dir', type=str, default='train_stage3_virchow2_clammb')
    parser.add_argument('--conch_dir', type=str, default='train_stage3_conch_clammb')
    parser.add_argument('--conch1_5_dir', type=str, default='train_stage3_conch1_5_clammb')
    parser.add_argument('--hoptimus_dir', type=str, default='train_stage3_h_optimus_1_clammb')
    parser.add_argument('--output_dir', type=str, default='evaluation_stage4')
    parser.add_argument('--track', type=str, default='score', choices=['score', 'map'],
                        help="选择评测的权重轨道: score (默认) 或 map")
    parser.add_argument('--gigapath_dir', type=str, default='train_stage3_gigapath_clammb')
    parser.add_argument('--allow_partial_oof', action='store_true', default=False,
                        help="允许不足 5 折的不完整模型参与 OOF 样本对齐与权重搜索 (默认严格拦截以防样本量缩水)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.csv_path)
    test_df = df[df['fold'].astype(str).str.strip().str.lower() == 'test'].reset_index(drop=True)
    if len(test_df) == 0:
        print("⚠️ 警告: 未找到 fold=='test'，自动回退使用全量数据进行评估！")
        test_df = df

    print(f"独立测试集样本数: {len(test_df)} 例")

    label_cols = [c for c in df.columns if c.startswith('label_')]
    targets_main = test_df[label_cols].fillna(0).values.astype(np.float32)

    # 防御性提取 DCIS 亚型子标签 (若列缺失则安全以 0 填充，杜绝盲测集 KeyError)
    dcis_type_cols = ['dcis_type_solid', 'dcis_type_cribriform', 'dcis_type_micropapillary']
    dcis_types_mat = np.zeros((len(test_df), len(dcis_type_cols)), dtype=int)
    for c_idx, c_col in enumerate(dcis_type_cols):
        if c_col in test_df.columns:
            dcis_types_mat[:, c_idx] = test_df[c_col].fillna(0).astype(int).values

    # 提取测试集子任务真实标签 (完整闭环 NST 三细项与 DCIS 分级，强健解析 float 字符串)
    def _s_int(v, d=-1):
        if pd.isna(v):
            return d
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return d
    aux_targets = {
        'nst_tubule': np.array([_s_int(x) - 1 if _s_int(x) > 0 else -1 for x in test_df.get('nst_grade_tubule', pd.Series([-1]*len(test_df)))]),
        'nst_nuclear': np.array([_s_int(x) - 1 if _s_int(x) > 0 else -1 for x in test_df.get('nst_grade_nuclear', pd.Series([-1]*len(test_df)))]),
        'nst_mitoses': np.array([_s_int(x) - 1 if _s_int(x) > 0 else -1 for x in test_df.get('nst_grade_mitoses', pd.Series([-1]*len(test_df)))]),
        'nst_overall': np.array([_s_int(x) - 1 if _s_int(x) > 0 else -1 for x in test_df.get('nst_grade_overall', pd.Series([-1]*len(test_df)))]),
        'dcis_grade': np.array([_s_int(x) - 1 if _s_int(x) > 0 else -1 for x in test_df.get('dcis_grade', pd.Series([-1]*len(test_df)))]),
        'dcis_necrosis': np.array([_s_int(x) for x in test_df.get('dcis_necrosis', pd.Series([-1]*len(test_df)))]),
        'dcis_types': dcis_types_mat
    }

    # 提前锁定原生测试集文件名
    raw_test_ids = test_df['id'].astype(str).tolist() if 'id' in test_df.columns else (
        test_df['wsi_id'].astype(str).tolist() if 'wsi_id' in test_df.columns else [f"slide_{i}.tiff" for i in range(len(test_df))]
    )

    # 动态确定执行轨道：传入 map 时自动双轨并行，否则仅执行 score 轨
    tracks_to_run = ['score', 'map'] if args.track == 'map' else ['score']

    for cur_track in tracks_to_run:
        # 双轨模式下自动建子目录分流保存，单轨模式直接输出到 output_dir
        cur_out_dir = os.path.join(args.output_dir, cur_track) if len(tracks_to_run) > 1 else args.output_dir
        os.makedirs(cur_out_dir, exist_ok=True)

        print(f"\n{'='*30} 启动 Stage 4 集成评测系统 [轨道: {cur_track.upper()}] {'='*30}")

        # 1. 独立运行三大模型 5 折推断 (增强动态热插拔与权重检测)
        predictions = {}
        confirmed_model_dirs = {}
        backbones_config = [
            ('CONCH', args.conch_dir, 'conch', 512),
            ('CONCH1.5', args.conch1_5_dir, 'conch1_5', 1024),
            ('Virchow2', args.virchow2_dir, 'virchow2', 1280),
            ('H-optimus-1', args.hoptimus_dir, 'h_optimus_1', 1536),
            ('GigaPath', args.gigapath_dir, 'gigapath', 1536)
        ]

        for name, m_dir, col_prefix, dim in backbones_config:
            # 精确适配目录名下划线与短横线变体
            candidates = [
                m_dir,
                m_dir.replace('h_optimus_1', 'h-optimus-1') if 'h_optimus_1' in m_dir else m_dir,
                m_dir.replace('h-optimus-1', 'h_optimus_1') if 'h-optimus-1' in m_dir else m_dir,
                m_dir.replace('conch1_5', 'conch-1-5') if 'conch1_5' in m_dir else m_dir,
                m_dir.replace('conch1_5', 'conch_1_5') if 'conch1_5' in m_dir else m_dir,
                m_dir.replace('conch1_5', 'conch1.5') if 'conch1_5' in m_dir else m_dir,
            ]
            candidates = list(dict.fromkeys(candidates))

            target_dir = None
            for c_dir in candidates:
                if _has_valid_checkpoints(c_dir, track=cur_track):
                    target_dir = c_dir
                    break

            if target_dir is not None:
                avail_folds = _get_available_folds(target_dir, track=cur_track)
                if len(avail_folds) < 5:
                    missing = [f for f in range(5) if f not in avail_folds]
                    warnings.warn(f"⚠️ [5折完整性警报] {name}: 仅检测到 {len(avail_folds)}/5 折权重 (已就绪: {avail_folds}, 缺失: {missing})，模型可能仍在训练中！")
                try:
                    preds = predict_backbone_5fold(target_dir, col_prefix, dim, test_df, label_cols, track=cur_track)
                    predictions[name] = preds
                    confirmed_model_dirs[name] = target_dir
                except Exception as e:
                    warnings.warn(f"⚠️ [模型热插拔] {name} 载入或推理失败，已安全降级跳过: {e}")
            else:
                dir_exists = any(os.path.isdir(c) for c in candidates)
                if dir_exists:
                    print(f"⚠️ [模型热插拔跳过] {name}: 目录存在但 checkpoints/ 中暂无有效权重 ({cur_track} 轨)")
                else:
                    print(f"⚠️ [跳过] {name}: 目录不存在 ({m_dir})")

        if not predictions:
            print(f"❌ 错误: [{cur_track.upper()} 轨] 未找到任何已训练的模型目录，请检查路径。")
            continue

        # 载入所有已训练模型的 OOF 预测数据 (使用已确认存在的路径，默认开启 5 折完整性校验)
        oof_data = load_all_oof_predictions(confirmed_model_dirs, strict_5fold=not args.allow_partial_oof, track=cur_track)
        names = list(predictions.keys())

        # 2. 构造对比实验方案 (全阶梯动态展开: 1模 -> 2模 -> 3模 -> ... -> N模全量)
        from itertools import combinations
        combos = {}

        def _calc_equal_ths(model_keys):
            """为任意阶梯的等权组合安全计算 OOF 标定阈值"""
            if not all(k in oof_data and 'probs' in oof_data[k] and len(oof_data[k]['probs']) > 0 for k in model_keys):
                return None
            min_r = min(min(len(oof_data[k]['probs']) for k in model_keys), len(oof_data[model_keys[0]]['targets']))
            if min_r == 0:
                return None
            eq_p = sum(oof_data[k]['probs'][:min_r] for k in model_keys) / len(model_keys)
            return calibrate_thresholds(oof_data[model_keys[0]]['targets'][:min_r], eq_p)

        # 动态遍历阶梯：k 从 1 到 N
        for k in range(1, len(names) + 1):
            for combo in combinations(names, k):
                model_keys = list(combo)
                tag = " + ".join(model_keys)

                if k == 1:
                    # 单模型基线
                    m_name = model_keys[0]
                    th = calibrate_thresholds(oof_data[m_name]['targets'], oof_data[m_name]['probs']) if m_name in oof_data else predictions[m_name]['thresholds']
                    combos[f"Single: {m_name}"] = (model_keys, [1.0], th)
                else:
                    prefix = "Pair" if k == 2 else ("Unified" if k == len(names) else f"Combo_{k}")
                    # 1) 全阶梯通用等权基线 (1/k)
                    combos[f"{prefix}: {tag} (Equal)"] = (model_keys, [1.0 / k] * k, _calc_equal_ths(model_keys))

                    # 2) 针对 2 模与 3 模自动搜索 OOF 最优加权
                    if k in [2, 3]:
                        w_opt, th_opt = optimize_weights_and_thresholds(oof_data, model_keys, track=cur_track)
                        if w_opt is not None and len(set(w_opt)) > 1:
                            w_str = ":".join([str(round(w, 2)) for w in w_opt])
                            combos[f"{prefix}: {tag} (OOF-Weighted {w_str})"] = (model_keys, w_opt, th_opt)

        # 3. 遍历计算各方案指标并收集报告
        summary_results = []
        combo_predictions_cache = {}

        for combo_name, (model_keys, weights, custom_ths) in combos.items():
            if weights is None:
                w = [1.0 / len(model_keys)] * len(model_keys)
            else:
                w = [x / sum(weights) for x in weights]

            # 概率加权融合
            ens_main = sum(predictions[k]['probs_main'] * w[i] for i, k in enumerate(model_keys))
            
            # 阈值决策: 优先使用 OOF 标定的集成阈值，未搜索方案回退到线性加权
            if custom_ths is not None:
                ens_th = custom_ths
            else:
                ens_th = sum(predictions[k]['thresholds'] * w[i] for i, k in enumerate(model_keys))
            
            ens_subs = {}
            for sub_k in predictions[model_keys[0]]['subtasks']:
                ens_subs[sub_k] = sum(predictions[k]['subtasks'][sub_k] * w[i] for i, k in enumerate(model_keys))

            metrics = compute_metrics(ens_main, ens_subs, targets_main, aux_targets, ens_th, label_cols)
            category = "Single" if len(model_keys) == 1 else ("Pairwise" if len(model_keys) == 2 else ("Unified" if len(model_keys) == len(names) else f"Combo_{len(model_keys)}"))
            metrics['Tier'] = category
            metrics['Experiment'] = combo_name
            # 综合分动态对齐轨道
            metrics['Score'] = metrics['mAP'] if cur_track == 'map' else round(0.7 * metrics['mAP'] + 0.3 * metrics['F1@Opt'], 4)
            metrics['Num_Models'] = len(model_keys) * 5
            summary_results.append(metrics)

            combo_predictions_cache[combo_name] = (ens_main, ens_subs, ens_th)

        # 4. 输出格式化控制台看板与 CSV 报告
        per_class_all = {res['Experiment']: res.pop('_per_class_f1') for res in summary_results}
        df_report = pd.DataFrame(summary_results)

        cols_order = [
            'Tier', 'Experiment', 'Score', 'mAP', 'AUROC', 'F1@0.5', 'F1@Opt',
            'NST_Tubule_F1', 'NST_Nuclear_F1', 'NST_Mitoses_F1', 'NST_Overall_F1',
            'DCIS_Grade_F1', 'DCIS_Necrosis_F1', 'DCIS_Type_F1', 'Num_Models'
        ]
        # 根据当前轨道决定首要排序列
        sort_key = 'mAP' if cur_track == 'map' else 'Score'
        df_report = df_report[cols_order].sort_values(by=sort_key, ascending=False)

        # 配置终端自适应超宽输出，杜绝 15 列横向折行打乱排版
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', None)

        print("\n" + "="*165)
        print(f"🏆 【独立测试集】单模型 / 两两配对 / 三模型大一统 综合评测总看板 [轨道: {cur_track.upper()}] (Holdout Test N={len(test_df)})")
        print("="*165)
        print(df_report.to_string(index=False, line_width=1000))
        print("="*165)

        # 导出阶梯式汇总表
        csv_save_path = os.path.join(cur_out_dir, 'ensemble_comparison_report.csv')
        df_report.to_csv(csv_save_path, index=False)
        print(f"\n✅ 对比汇总报告已导出至: {csv_save_path}")

        # 【新增】导出 22 维全标签细粒度 F1 横向进化对比表
        df_per_cls = pd.DataFrame(per_class_all).T
        df_per_cls.index.name = 'Experiment'
        per_cls_csv_path = os.path.join(cur_out_dir, 'per_class_f1_breakdown.csv')
        df_per_cls.to_csv(per_cls_csv_path)
        print(f"📊 22 维全类别 F1 细粒度演进报表已导出至: {per_cls_csv_path}")

        # 【新增】按梯队提炼阶梯式演进对比
        print(f"\n📊 --- [{cur_track.upper()} 轨] 模型进化阶梯分析 (各阶段最佳表现) ---")
        tiers_order = ['Single', 'Pairwise'] + [f'Combo_{k}' for k in range(3, len(names))] + ['Unified']
        seen_tiers = set()
        for tier in tiers_order:
            if tier in seen_tiers:
                continue
            seen_tiers.add(tier)
            tier_df = df_report[df_report['Tier'] == tier]
            if not tier_df.empty:
                best_tier = tier_df.iloc[0]
                print(f"[{tier:<8}] 最佳方案: {best_tier['Experiment']:<45} | Score: {best_tier['Score']:.4f} | mAP: {best_tier['mAP']:.4f} | F1@Opt: {best_tier['F1@Opt']:.4f}")

        # 辅助缓存：按模型组合独立生成 OOF 条件标定参数
        combo_calib_cache = {}
        def _get_combo_params(combo_n):
            if combo_n in combo_calib_cache:
                return combo_calib_cache[combo_n]
            m_keys, m_weights, c_ths = combos[combo_n]
            if all(k in oof_data for k in m_keys):
                m_r = min(len(oof_data[k]['probs']) for k in m_keys)
                norm_w = [x / sum(m_weights) for x in m_weights]
                f_p = sum(oof_data[k]['probs'][:m_r] * norm_w[i] for i, k in enumerate(m_keys))
                f_t = oof_data[m_keys[0]]['targets'][:m_r]
                base_th = c_ths if c_ths is not None else combo_predictions_cache[combo_n][2]
                p_dict = calibrate_conditional_thresholds(f_t, f_p, base_th, label_cols)
            else:
                p_dict = {'cond_thresholds': {}, 'beta': 0.25}
            combo_calib_cache[combo_n] = p_dict
            return p_dict

        def save_predictions_and_reports(combo_name, file_prefix):
            main_p, sub_p, th_v = combo_predictions_cache[combo_name]
            cur_params = _get_combo_params(combo_name)  # 获取该方案专属的 OOF 标定参数
            
            df_export = pd.DataFrame({'id': raw_test_ids})
            for idx, col in enumerate(label_cols):
                df_export[f"prob_{col}"] = np.round(main_p[:, idx], 4)
                df_export[f"pred_{col}"] = (main_p[:, idx] > th_v[idx]).astype(int)
            csv_p = os.path.join(cur_out_dir, f'test_predictions_{file_prefix}.csv')
            df_export.to_csv(csv_p, index=False)
            
            json_list = []
            for i in range(len(test_df)):
                sub_s = {k: sub_p[k][i] for k in sub_p}
                rep = generate_report_text(main_p[i], sub_s, th_v, label_cols, cascade_params=cur_params)
                json_list.append({"id": raw_test_ids[i], "report": rep})
            json_p = os.path.join(cur_out_dir, f'submission_{file_prefix}.json')
            with open(json_p, 'w', encoding='utf-8') as f:
                json.dump(json_list, f, indent=2, ensure_ascii=False)
                
            print(f"  📁 [{file_prefix}] 方案: {combo_name:<40} -> 已固化 CSV 与 JSON 报告")

        print(f"\n>>> [{cur_track.upper()} 轨] 开始全阶梯生成最终交付物 (单模型 / 各阶梯最优 / 全模大一统):")

        # (1) 导出每一个单模型的独立预测与报告
        for name in names:
            single_key = f"Single: {name}"
            if single_key in combo_predictions_cache:
                clean_name = name.lower().replace('-', '_')
                save_predictions_and_reports(single_key, f'single_{clean_name}')

        # (2) 动态遍历所有多模阶梯 (Pairwise, Combo_3, ..., Unified)，自动导出各阶梯最强方案
        for tier in [t for t in tiers_order if t != 'Single']:
            tier_df = df_report[df_report['Tier'] == tier]
            if not tier_df.empty:
                best_in_tier = tier_df.iloc[0]['Experiment']
                save_predictions_and_reports(best_in_tier, f'{tier.lower()}_best')

        # (3) 导出全局总冠军方案 (首选最优方案)
        best_overall_name = df_report.iloc[0]['Experiment']
        save_predictions_and_reports(best_overall_name, f'ensemble_{cur_track}')
        print(f"\n🏆 [{cur_track.upper()} 轨] 全局总冠军方案为: [{best_overall_name}] (交付文件: submission_ensemble_{cur_track}.json)")

if __name__ == "__main__":
    main()