import pandas as pd
import numpy as np
import os
import argparse
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Merge Labels with Dual-Model Features and Stratify")
    # 默认路径绑定至脚本同级目录
    parser.add_argument('--label_path', type=Path, default=script_dir / 'breast_cancer_multilabel_targets.csv', 
                        help='Path to label CSV')
    parser.add_argument('--features_root', type=Path, default=Path('/mnt/e/Extracted_Features_train_20x'), 
                        help='Root directory containing virchow2 and conch features')
    parser.add_argument('--organ', type=str, default='Breast', help='Subfolder organ name')
    parser.add_argument('--output_path', type=Path, default=script_dir / 'final_train_list_multilabel.csv', 
                        help='Path to save final list')
    parser.add_argument('--downsample_factor', type=int, default=1, help='Downsample factor used in Stage 1 (1 for 20x)')
    parser.add_argument('--n_splits', type=int, default=6, help='Number of CV splits (1 test + 5 train/val)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for stratification')
    args = parser.parse_args()

    if not args.label_path.exists():
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

    # 2. 检查双模型特征及坐标文件是否存在 (双模交集清洗)
    virchow2_dir = Path(args.features_root) / "virchow2" / args.organ
    conch_dir = Path(args.features_root) / "conch" / args.organ

    valid_data = []
    print(f"正在对齐并检查双模型特征文件:\n - Virchow2 目录: {virchow2_dir}\n - CONCH 目录: {conch_dir}")
    
    found_count = 0
    missing_count = 0
    ds = args.downsample_factor
    
    for idx, row in df.iterrows():
        wsi_id_full = str(row['id'])
        wsi_id = Path(wsi_id_full).stem  # 安全去除 .tiff / .svs 后缀
        
        # 构造预期的三个核心文件路径
        v2_feat = virchow2_dir / f"{wsi_id}_features_downsampled{ds}x.npy"
        conch_feat = conch_dir / f"{wsi_id}_features_downsampled{ds}x.npy"
        
        # 坐标优先取 virchow2 下的，若无则取 conch 下的
        coord_file = virchow2_dir / f"{wsi_id}_coords_downsampled{ds}x.npy"
        if not coord_file.exists():
            coord_file = conch_dir / f"{wsi_id}_coords_downsampled{ds}x.npy"

        # 【核心约束】只有当双特征和坐标完全齐备时，才纳入有效数据集（保障后续 Avg-Pred 100% 对齐）
        if v2_feat.exists() and conch_feat.exists() and coord_file.exists():
            row_dict = row.to_dict()
            row_dict['wsi_id'] = wsi_id
            row_dict['virchow2_feature_path'] = str(v2_feat)
            row_dict['conch_feature_path'] = str(conch_feat)
            row_dict['coord_path'] = str(coord_file)
            
            valid_data.append(row_dict)
            found_count += 1
        else:
            missing_count += 1
            if missing_count <= 3:
                missing_info = []
                if not v2_feat.exists(): missing_info.append("Virchow2缺失")
                if not conch_feat.exists(): missing_info.append("CONCH缺失")
                if not coord_file.exists(): missing_info.append("Coords缺失")
                print(f"Warning: 样本 {wsi_id} 文件不完整: {', '.join(missing_info)}")
        
    if not valid_data:
        print("❌ 错误: 没有找到任何满足双特征齐备的样本！请检查 Stage 1 是否已提取完毕。")
        return
    
    print(f"✅ 双特征交集对齐完成: 成功 {found_count} 例, 缺失或不全 {missing_count} 例")
    df_valid = pd.DataFrame(valid_data)

    # 3. 执行真正的多标签分层切分 (Multilabel Stratified Split)
    label_cols = [c for c in df_valid.columns if c.startswith('label_')]
    Y_multilabel = df_valid[label_cols].fillna(0).values.astype(int)
    X_dummy = np.zeros(len(df_valid))

    mskf = MultilabelStratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    
    df_valid['fold'] = -1
    print(f"开始基于 {len(label_cols)} 个标签的多标签分层抽样 (n_splits={args.n_splits}, seed={args.seed})...")
    
    for fold_idx, (train_index, test_index) in enumerate(mskf.split(X_dummy, Y_multilabel)):
        df_valid.iloc[test_index, df_valid.columns.get_loc('fold')] = fold_idx
        
    # 4. 重命名 Fold (0 -> test, 1-5 -> '0'-'4')
    def remap_fold(x):
        return 'test' if x == 0 else str(x - 1)
            
    df_valid['fold'] = df_valid['fold'].apply(remap_fold)

    # 5. 详细分布检查 (包含新加入的子标签 + 稀有类别检查)
    print("\n--- 分布检查 ---")
    print("Fold 分布:")
    print(df_valid['fold'].value_counts())
    
    # 检查新增的临床关键分级 (验证 Nottingham 3级与 DCIS 核分级)
    if 'nst_grade_overall' in df_valid.columns:
        print("\n>> Nottingham Overall Grade (1=G1, 2=G2, 3=G3, -1=未提及) 分布:")
        print(df_valid['nst_grade_overall'].value_counts())

    if 'dcis_grade' in df_valid.columns:
        print("\n>> DCIS Grade (1=Low, 2=Intermediate, 3=High, -1=未提及) 分布:")
        print(df_valid['dcis_grade'].value_counts())

    # 【重要】更新为 22 维体系下真实的极长尾稀有类别 (样本数 <= 11 例)
    print("\n--- [重要] 22维体系稀有类别在各 Fold 的均衡性检查 ---")
    rare_checks = [
        'label_lymphoma',          # 恶性淋巴瘤 (6例)
        'label_micro_invasive',    # 微浸润癌 (6例)
        'label_fea',               # 平坦上皮不典型增生 (6例)
        'label_micropapillary',    # 浸润性微乳头状癌 (9例)
        'label_phyllodes'          # 叶状肿瘤 (11例)
    ]
    
    for label in rare_checks:
        if label in df_valid.columns:
            subset = df_valid[df_valid[label] == 1]
            count = len(subset)
            print(f"\n>> {label} (总数: {count}) 在各 Fold 的分布:")
            print(subset['fold'].value_counts())
            
            # 严格核查测试集 (Test) 是否有代表样本
            in_test = len(subset[subset['fold'] == 'test'])
            print(f"   -> 测试集 (Test) 分得: {in_test} 例 | 交叉验证集 (0~4) 分得: {count - in_test} 例")

    # 6. 保存
    # 直接保存，无需 drop
    df_valid.to_csv(args.output_path, index=False)
    print(f"\n✅ 最终列表已生成: {args.output_path}")
    
if __name__ == "__main__":
    main()