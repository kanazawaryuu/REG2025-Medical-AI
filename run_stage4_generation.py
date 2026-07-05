import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import json
import re

# --- 1. 配置区域 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 模型定义 (必须与 Stage 3.5 训练时的架构完全一致: MLP Head) ---

class PredictionHead(nn.Module):
    """
    MLP 预测头：Linear -> ReLU -> Dropout -> Linear
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
        
        # --- 多任务头 (与训练代码一致) ---
        self.head_main = nn.Linear(hidden_dim, n_classes) # 主任务
        
        # NST 子任务 (3分类)
        self.head_nst_tubule = PredictionHead(hidden_dim, 3)
        self.head_nst_nuclear = PredictionHead(hidden_dim, 3)
        self.head_nst_mitoses = PredictionHead(hidden_dim, 3)
        
        # DCIS 子任务
        self.head_dcis_grade = PredictionHead(hidden_dim, 3)    
        self.head_dcis_necrosis = PredictionHead(hidden_dim, 3) 
        self.head_dcis_types = PredictionHead(hidden_dim, 3)    

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor(x)
        A = self.attention_weights(self.attention_V(H) * self.attention_U(H))
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)
        
        # 返回字典，包含所有头的输出 logits
        outputs = {
            'main': self.head_main(M),
            'nst_tubule': self.head_nst_tubule(M),
            'nst_nuclear': self.head_nst_nuclear(M),
            'nst_mitoses': self.head_nst_mitoses(M),
            'dcis_grade': self.head_dcis_grade(M),
            'dcis_necrosis': self.head_dcis_necrosis(M),
            'dcis_types': self.head_dcis_types(M)
        }
        return outputs

# 数据加载器 (保持不变，略作优化)
class InferenceDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_path = row['feature_path']
        wsi_id = row['id']
        
        if not os.path.exists(feat_path):
            return wsi_id, torch.zeros((1, 1536)).float()
            
        try:
            features = np.load(feat_path)
            if features.shape[0] > 12000: 
                 features = features[:12000]
            features = torch.from_numpy(features).float()
        except:
            features = torch.zeros((1, 1536)).float()
            
        return wsi_id, features

# --- 2. 核心功能函数 ---

def load_ensemble_models(model_dir, n_classes):
    """加载所有 5 折的模型"""
    models = []
    print(f"正在加载集成模型 (5-Fold Ensemble)...")
    for fold in range(5):
        path = os.path.join(model_dir, 'checkpoints', f'best_model_fold{fold}.pth')
        if not os.path.exists(path):
            print(f"⚠️ 警告: 缺失 Fold {fold} 的模型 ({path})，将跳过。")
            continue
            
        model = GatedAttentionMIL(n_classes=n_classes).to(DEVICE)
        # weights_only=True 是为了消除警告，如果报错可改为 False
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        except:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            
        model.eval()
        models.append(model)
    
    if len(models) == 0:
        raise FileNotFoundError("❌ 未找到任何模型文件！请检查路径。")
        
    print(f"✅ 成功加载 {len(models)} 个模型。")
    return models

def load_strategic_thresholds(model_dir, label_cols):
    """
    加载阈值 (禁用激进召回，完全信赖验证集数据)
    """
    final_thresholds = {}
    
    # 读取 5 折算出的阈值
    fold_ths = {col: [] for col in label_cols}
    
    for fold in range(5):
        path = os.path.join(model_dir, 'metadata', f'thresholds_fold{fold}.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    cls, th = line.strip().split(',')
                    fold_ths[cls].append(float(th))
    
    print("\n--- [Stage 4 阈值策略配置 (保守模式)] ---")
    for col in label_cols:
        if not fold_ths[col]:
            avg_th = 0.5
        else:
            # 直接使用验证集算出的最佳阈值平均值
            avg_th = np.mean(fold_ths[col])
        
        final_th = avg_th
        
        # --- 策略修改：注释掉激进召回逻辑 ---
        # 只要把下面这些 if...elif 注释掉，就是"原汁原味"的模型表现
        
        # ultra_rare = ['label_metaplastic', 'label_micropapillary', ...]
        # if col in ultra_rare:
        #     final_th = min(avg_th, 0.15) 
        # elif col in ['label_mucinous', 'label_invasive_lobular']:
        #     final_th = min(avg_th, 0.4)
            
        print(f"🔒 锁定阈值: {col:<25} 设定={final_th:.2f}")
            
        final_thresholds[col] = final_th
        
    return final_thresholds

# --- 3. 核心报告生成逻辑 (Part 2) ---

def generate_report_text(avg_outputs, thresholds, label_cols):
    """
    【生成核心】将模型输出转换为符合病理医生习惯的文本报告
    avg_outputs: 包含 'main' 及所有子任务头平均概率/logits 的字典
    thresholds: 各类别的最佳二值化阈值
    label_cols: 标签列名列表
    """
    
    # 1. 提取主要预测
    main_probs = avg_outputs['main'].cpu().numpy()[0]
    
   # --- 1. 词汇映射 (基于统计数据的 Top 1 词汇) ---
    text_map = {
        # 浸润性癌
        'label_invasive_nst': "Invasive carcinoma of no special type", # 放弃 Grade，保准确率
        'label_invasive_lobular': "Invasive lobular carcinoma",
        'label_mucinous': "Mucinous carcinoma",
        'label_micropapillary': "Invasive micropapillary carcinoma",
        'label_metaplastic': "Metaplastic carcinoma",
        'label_micro_invasive': "Micro-invasive carcinoma",
        'label_cribriform_invasive': "Invasive cribriform carcinoma",
        'label_tubular_carcinoma': "Tubular carcinoma",
        
        # 原位癌
        'label_dcis': "Ductal carcinoma in situ",
        'label_lcis': "Lobular carcinoma in situ",
        
        # 良性/交界性
        'label_fibroepithelial': "Fibroadenoma", # [统计] 90% 的样本写的是这个，不是 Fibroepithelial lesion
        'label_fibroadenoma': "Fibroadenoma",    # [统计] 保持一致，虽然有长写法，但短写法更通用
        'label_phyllodes': "Phyllodes tumor",    # [统计] 36% 是这个，63% 是 favor phyllodes，选短的
        'label_papillary_neoplasm': "Papillary neoplasm", # [统计] 66% 命中率，改回 Neoplasm
        'label_papilloma': "Intraductal papilloma",       # [统计] 45% 命中率
        
        # 增生与癌前病变
        'label_adh': "Atypical ductal hyperplasia",
        'label_alh': "Atypical lobular hyperplasia",
        'label_fea': "Flat epithelial atypia",
        'label_udh': "Usual ductal hyperplasia",
        'label_columnar': "Columnar cell lesion", # [统计] 91.8% 是 lesion，不是 change
        'label_sclerosing_adenosis': "Sclerosing adenosis",
        
        # 其他
        'label_lymphoma': "Malignant lymphoma",
        'label_microcalcification': "Microcalcification",
        # 【关键修正】统计显示 91.7% 是 No evidence of tumor
        'label_no_tumor': "No evidence of tumor" 
    }

    # --- 2. 伴随特征映射 ---
    feature_map = {
        'label_mucinous': "Mucinous features",
        'label_micropapillary': "Micropapillary differentiation",
        'label_metaplastic': "Metaplastic differentiation",
        'label_tubular_carcinoma': "Tubular features",
        'label_cribriform_invasive': "Cribriform pattern",
        # [统计] 21.7% 的 Fibroadenoma 标签下出现了 Fibroadenomatoid change
        'label_fibroadenoma': "Fibroadenomatoid change", 
        'label_microcalcification': "Microcalcification",
        'label_columnar': "Columnar cell change", # 作为特征时，change 也很常见
        'label_sclerosing_adenosis': "Sclerosing adenosis",
    }

    # 3. 确定主诊断 (Primary Diagnosis)
    # 排除仅作为特征出现的标签
    secondary_only_labels = ['label_microcalcification']
    max_prob = -1
    primary_label = None
    
    for i, cls in enumerate(label_cols):
        if cls in secondary_only_labels: continue
        p = main_probs[i]
        
        # 使用特定的阈值判断是否激活，然后取最高分
        # 注意：这里我们主要用 max_prob 找最大类，但必须参考 threshold 过滤低置信度
        if p > max_prob:
            max_prob = p
            primary_label = cls
            
    # --- 兜底逻辑 ---
    no_tumor_idx = label_cols.index('label_no_tumor') if 'label_no_tumor' in label_cols else -1
    
    # 策略 A: 优先信赖 No Tumor (如果分高且过线)
    if no_tumor_idx != -1 and main_probs[no_tumor_idx] > max_prob and main_probs[no_tumor_idx] > 0.5:
        primary_label = 'label_no_tumor'
    
    # 策略 B: 如果最高分都很低 (<0.3)，且不是 No Tumor，强行归类为 NST (最常见)
    elif max_prob < 0.3 and primary_label != 'label_no_tumor':
        primary_label = 'label_invasive_nst'

    # 4. 生成主诊断文本 (结合子任务)
    findings = []
    primary_text = text_map.get(primary_label, "Lesion")
    
    # === 逻辑分支：NST 动态分级 ===
    if primary_label == 'label_invasive_nst':
        # 从子任务头获取分数 (argmax + 1) -> 范围 1,2,3
        # 使用 5 折平均后的 logits 直接取最大值，非常稳健
        t_score = avg_outputs['nst_tubule'].argmax(dim=1).item() + 1
        n_score = avg_outputs['nst_nuclear'].argmax(dim=1).item() + 1
        m_score = avg_outputs['nst_mitoses'].argmax(dim=1).item() + 1
        
        # 计算总分与 Grade
        total_score = t_score + n_score + m_score
        if total_score <= 5: grade_roman = "I"
        elif total_score <= 7: grade_roman = "II"
        else: grade_roman = "III"
        
        # 格式化输出: Invasive carcinoma of no special type, grade II (Tubule formation: 3, Nuclear grade: 2, Mitoses: 1)
        primary_text = f"{primary_text}, grade {grade_roman} (Tubule formation: {t_score}, Nuclear grade: {n_score}, Mitoses: {m_score})"

    # === 逻辑分支：DCIS 详细描述 ===
    elif primary_label == 'label_dcis':
        # 1. 核分级
        g_idx = avg_outputs['dcis_grade'].argmax(dim=1).item()
        g_map = ["Low", "Intermediate", "High"]
        g_text = g_map[g_idx]
        
        # 2. 坏死
        n_idx = avg_outputs['dcis_necrosis'].argmax(dim=1).item()
        n_map = ["Absent", "Present", "Present"] # Focal 和 Comedo 都算 Present
        # 如果你想区分更细：
        n_map_detail = ["Absent", "Present (Focal)", "Present (Comedo-type)"]
        n_text = n_map_detail[n_idx]
        
        # 3. 结构类型 (取最大概率的一个)
        # 顺序对应: Solid, Cribriform, Micropapillary
        t_idx = avg_outputs['dcis_types'].argmax(dim=1).item()
        t_map = ["Solid", "Cribriform", "Micropapillary"]
        t_text = t_map[t_idx]
        
        # 格式化输出 (多行)
        # 注意格式：主诊断换行后跟子项
        primary_text = f"{primary_text}\n  - Type: {t_text}\n  - Nuclear grade: {g_text}\n  - Necrosis: {n_text}"

    findings.append(primary_text)

    # 5. 收集次要发现 (Secondary Findings)
    for i, col in enumerate(label_cols):
        if col == primary_label: continue
        if col == 'label_no_tumor': continue
        # 互斥逻辑
        if primary_label == 'label_fibroadenoma' and col == 'label_fibroepithelial': continue
        if primary_label == 'label_phyllodes' and col == 'label_fibroepithelial': continue
        if primary_label == 'label_papilloma' and col == 'label_papillary_neoplasm': continue 
        if primary_label == 'label_papillary_neoplasm' and col == 'label_papilloma': continue 
        if primary_label == 'label_columnar' and 'columnar' in col: continue

        p = main_probs[i]
        th = thresholds.get(col, 0.5)
        
        if p > th:
            # 如果是特征映射里有的，用特征描述；否则用通用描述
            desc = feature_map.get(col, text_map.get(col, col))
            findings.append(desc)

    # 6. 最终组装
    header = "Breast;"
    
    # 清理：如果判为 No Tumor 但又有其他发现，移除 No evidence
    if primary_label == 'label_no_tumor' and len(findings) > 1:
        findings.pop(0) # 移除 "No evidence of tumor"

    if len(findings) == 0:
        report = f"{header}\n  No evidence of tumor"
    elif len(findings) == 1:
        # 如果只有一行且包含换行符（如 DCIS），不用加 '1.'
        if "\n" in findings[0]: 
             report = f"{header}\n  {findings[0]}"
        else:
             report = f"{header}\n  {findings[0]}"
    else:
        lines = [header]
        for idx, item in enumerate(findings):
            # 处理多行子项的缩进 (DCIS)
            if "\n" in item:
                lines.append(f"  {idx + 1}. {item}")
            else:
                lines.append(f"  {idx + 1}. {item}")
        report = "\n".join(lines)

    return report

# --- 4. 主执行函数 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='final_train_list_multilabel.csv')
    parser.add_argument('--model_dir', type=str, default='train_stage3_attention')
    parser.add_argument('--output_path', type=str, default='submission.csv')
    args = parser.parse_args()
    
    # 1. 准备数据
    df = pd.read_csv(args.csv_path)
    test_df = df[df['fold'] == 'test'].reset_index(drop=True)
    
    if len(test_df) == 0:
        print("⚠️ 警告: fold='test' 为空！使用所有数据测试 (Debug)...")
        test_df = df
        
    print(f"待推理样本数: {len(test_df)}")
    
    label_cols = [c for c in df.columns if c.startswith('label_')]
    n_classes = len(label_cols)
    
    dataset = InferenceDataset(test_df)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
    
    # 2. 加载模型与阈值
    models = load_ensemble_models(args.model_dir, n_classes)
    thresholds = load_strategic_thresholds(args.model_dir, label_cols)
    
    # 3. 推理循环
    results = [] 
    
    print(">>> 开始推理 (Ensemble Inference)...")
    with torch.no_grad():
        for wsi_id, features in tqdm(loader):
            features = features.to(DEVICE)
            
            # 初始化累加器字典
            avg_outputs = {
                'main': torch.zeros(1, n_classes).to(DEVICE),
                'nst_tubule': torch.zeros(1, 3).to(DEVICE),
                'nst_nuclear': torch.zeros(1, 3).to(DEVICE),
                'nst_mitoses': torch.zeros(1, 3).to(DEVICE),
                'dcis_grade': torch.zeros(1, 3).to(DEVICE),
                'dcis_necrosis': torch.zeros(1, 3).to(DEVICE),
                'dcis_types': torch.zeros(1, 3).to(DEVICE)
            }
            
            # 5折集成
            for model in models:
                outputs = model(features)
                
                # 主任务：累加 Sigmoid 概率
                avg_outputs['main'] += torch.sigmoid(outputs['main'])
                
                # 子任务：累加 Softmax 概率 (比累加 logits 更平滑)
                avg_outputs['nst_tubule'] += torch.softmax(outputs['nst_tubule'], dim=1)
                avg_outputs['nst_nuclear'] += torch.softmax(outputs['nst_nuclear'], dim=1)
                avg_outputs['nst_mitoses'] += torch.softmax(outputs['nst_mitoses'], dim=1)
                
                avg_outputs['dcis_grade'] += torch.softmax(outputs['dcis_grade'], dim=1)
                avg_outputs['dcis_necrosis'] += torch.softmax(outputs['dcis_necrosis'], dim=1)
                # DCIS Type 也是 3分类头
                avg_outputs['dcis_types'] += torch.softmax(outputs['dcis_types'], dim=1)

            # 取平均
            for k in avg_outputs:
                avg_outputs[k] /= len(models)
            
            # 生成报告
            # 注意：wsi_id 是 tuple ('XXX',)，取 [0]
            report_text = generate_report_text(avg_outputs, thresholds, label_cols)
            
            entry = {
                "id": wsi_id[0], 
                "report": report_text
            }
            results.append(entry)
            
    # 4. 保存 JSON
    json_output_path = args.output_path.replace('.csv', '.json')
    if not json_output_path.endswith('.json'):
        json_output_path += '.json'
        
    print(f"正在保存 JSON 文件...")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 推理完成！标准格式结果已保存至: {json_output_path}")

if __name__ == "__main__":
    main()