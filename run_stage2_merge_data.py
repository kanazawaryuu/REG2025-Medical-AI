import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

# --- (中间的 get_stratification_label 函数保持不变) ---
# --- 配置区域 ---
FEATURES_DIR = os.path.expanduser("~/projects/Extracted_Features_train_CLEANED_all_tiles/Breast")
LABELS_CSV = "breast_cancer_multilabel_targets.csv"     # Stage 2 生成的标签
OUTPUT_LIST = "final_train_list_multilabel.csv"         # 最终生成的列表
DOWNSAMPLE_FACTOR = 2
N_SPLITS = 6  # 切成 6 份 (1份做测试，5份做训练)
SEED = 42     # 固定随机种子

def get_stratification_label(row):
    """
    为多标签样本定义一个'主标签'用于分层。
    逻辑：严格按照从【极度稀有】到【常见】的顺序判断。
    一旦匹配到稀有标签，立即返回，不再看后面的常见标签。
    """
    # 优先级列表：基于您的数据集统计 (越稀有越靠前)
    # 必须包含所有 24 个标签 + 阴性
    priority_labels = [
        # --- Tier 1: 极度稀有 (< 5 例) ---
        'label_alh',               # ~1 例 (非典型小叶增生) -> 最稀有！
        'label_tubular_carcinoma', # ~2 例
        'label_cribriform_invasive', # ~2 例
        'label_metaplastic',       # ~4 例
        
        # --- Tier 2: 非常稀有 (5 - 10 例) ---
        'label_micro_invasive',    # ~6 例
        'label_lymphoma',          # ~6 例
        'label_fea',               # ~6 例
        'label_micropapillary',    # ~8 例
        
        # --- Tier 3: 稀有 (10 - 50 例) ---
        'label_phyllodes',         # ~11 例
        'label_lcis',              # ~21 例
        'label_sclerosing_adenosis', # ~25 例
        'label_adh',               # ~36 例
        'label_papilloma',         # ~41 例 (这里指特指的导管内乳头状瘤)
        
        # --- Tier 4: 少见 (50 - 100 例) ---
        'label_columnar',          # ~50 例
        'label_mucinous',          # ~53 例 (我们重点关注的中等稀有类)
        'label_udh',               # ~58 例
        'label_fibroadenoma',      # ~69 例 (注意：需排在 fibroepithelial 之前)
        'label_invasive_lobular',  # ~78 例
        
        # --- Tier 5: 常见 (> 100 例) ---
        'label_fibroepithelial',   # ~135 例 (大类)
        'label_microcalcification',# ~161 例
        'label_papillary_neoplasm',# ~200 例 (大类)
        'label_dcis',              # ~451 例
        'label_invasive_nst',      # ~915 例 (最常见)
        
        # --- 兜底 ---
        'label_no_tumor'           # 阴性
    ]
    
    for label in priority_labels:
        # get(label, 0) 防止 CSV 里没有这一列报错
        if row.get(label, 0) == 1:
            return label
            
    return 'label_no_tumor' # 如果全为0，默认归为阴性

def main():
    parser = argparse.ArgumentParser(description="Merge Labels with Features and Stratify")
    # 默认值适配您的环境
    parser.add_argument('--label_path', type=str, default='breast_cancer_multilabel_targets.csv', help='Path to label CSV from Stage 2')
    parser.add_argument('--features_dir', type=str, default=os.path.expanduser("~/projects/Extracted_Features_train_CLEANED_all_tiles/Breast"), help='Directory containing .npy features')
    parser.add_argument('--output_path', type=str, default='final_train_list_multilabel.csv', help='Path to save final list')
    parser.add_argument('--n_splits', type=int, default=6, help='Number of CV splits')
    args = parser.parse_args()

    if not os.path.exists(args.label_path):
        print(f"❌ 错误: 找不到标签文件 {args.label_path}，请先运行 Stage 2 Parse Labels。")
        return

    # 1. 读取标签
    df = pd.read_csv(args.label_path)
    print(f"读取标签文件: {len(df)} 行")
    
    # 【关键检查】确认详细分级列是否存在
    # 这一步是为了防止 Stage 2 没跑对，导致重要的 grade 信息丢失
    detailed_cols = [c for c in df.columns if c.startswith('nst_grade') or c.startswith('dcis_')]
    print(f"✅ 检测到 {len(detailed_cols)} 个详细子标签列: {detailed_cols}")
    if len(detailed_cols) == 0:
        print("⚠️ 警告: 未检测到子标签 (nst_grade/dcis)。请确认是否运行了更新后的 Stage 2 脚本！")

    # 2. 检查特征文件是否存在 (数据清洗)
    valid_data = []
    print(f"正在目录 {args.features_dir} 中匹配特征文件...")
    
    # 统计找到和没找到的数量
    found_count = 0
    missing_count = 0
    
    for idx, row in df.iterrows():
        wsi_id_full = row['id']
        wsi_id = os.path.splitext(wsi_id_full)[0] # 去除 .tiff 后缀
        
        # 特征文件名格式
        fname = f"{wsi_id}_features_downsampled2x.npy" 
        final_path = os.path.join(args.features_dir, fname)

        if os.path.exists(final_path):
            row_dict = row.to_dict()
            row_dict['feature_path'] = final_path
            # 使用原来的逻辑生成分层用的主标签
            # 这样可以确保只根据 24 个主要病种进行分层，不受稀疏 Grade 标签的干扰
            row_dict['stratify_group'] = get_stratification_label(row)
            valid_data.append(row_dict)
            found_count += 1
        else:
            missing_count += 1
            if missing_count <= 3: # 只打印前3个缺失的作为示例
                print(f"Warning: 找不到文件: {final_path}")
        
    if not valid_data:
        print("❌ 错误: 没有找到任何匹配的特征文件！请检查路径配置。")
        return
    
    print(f"匹配完成: 成功 {found_count} 例, 缺失 {missing_count} 例")
    df_valid = pd.DataFrame(valid_data)

    # 3. 执行分层切分 (Stratified Split)
    # 策略: 依然基于 'stratify_group' (24大类) 进行分层
    
    # 【修改】生成随机种子，确保每次运行脚本时 Test 集都不一样
    # 使用 0~100000 之间的随机整数作为种子，并打印出来以便记录（如果这次跑分很高，你可以记下这个种子复现）
    current_seed = np.random.randint(0, 100000)
    print(f"🎲 本次随机分组种子 (Random Seed): {current_seed}")
    
    # 将生成的随机种子传入 random_state
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=current_seed)
    
    df_valid['fold'] = -1
    X_dummy = np.zeros(len(df_valid))
    y_stratify = df_valid['stratify_group']
    
    print(f"开始分层抽样 (n_splits={args.n_splits})...")
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X_dummy, y_stratify)):
        df_valid.iloc[test_index, df_valid.columns.get_loc('fold')] = fold_idx
        
    # 4. 重命名 Fold (0 -> test, 1-5 -> 0-4)
    def remap_fold(x):
        if x == 0: return 'test'
        else: return x - 1
            
    df_valid['fold'] = df_valid['fold'].apply(remap_fold)

# 5. 详细分布检查 (包含新加入的子标签 + 稀有类别检查)
    print("\n--- 分布检查 ---")
    print("Fold 分布:")
    print(df_valid['fold'].value_counts())
    
    # 检查新加入的子标签 (验证 Stage 2 的成果)
    if 'dcis_grade' in df_valid.columns:
        print("\n>> DCIS Grade 分布 (验证子标签是否保留):")
        print(df_valid['dcis_grade'].value_counts())

    # 【补回】稀有类别分布检查 (验证分层抽样的合理性)
    print("\n--- [重要] 稀有类别分布检查 ---")
    rare_checks = ['label_metaplastic', 'label_tubular_carcinoma', 'label_mucinous', 'label_micropapillary']
    
    for label in rare_checks:
        if label in df_valid.columns:
            subset = df_valid[df_valid[label] == 1]
            count = len(subset)
            print(f"\n>> {label} (总数: {count}) 在各 Fold 的分布:")
            print(subset['fold'].value_counts())
            
            # 检查测试集里有没有
            in_test = len(subset[subset['fold'] == 'test'])
            print(f"   -> 测试集 (Test) 包含: {in_test} 例")

    # 6. 保存
    # 移除辅助分层列，但保留所有原始标签(包括 nst_grade 等)
    df_valid.drop(columns=['stratify_group'], inplace=True)
    df_valid.to_csv(args.output_path, index=False)
    print(f"\n✅ 最终列表已生成: {args.output_path}")
    
if __name__ == "__main__":
    main()