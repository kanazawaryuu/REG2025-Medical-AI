import pandas as pd
import numpy as np
import os
import shutil
import argparse
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Merge Labels with Multi-Model Features and Stratify")
    # 默认路径绑定至脚本同级目录
    parser.add_argument('--label_path', type=Path, default=script_dir / 'breast_cancer_multilabel_targets.csv', 
                        help='Path to label CSV')
    parser.add_argument('--features_root', type=Path, default=Path('/mnt/e/Extracted_Features_train_20x'), 
                        help='Root directory containing virchow2, conch and h-optimus-1 features')
    parser.add_argument('--organ', type=str, default='Breast', help='Subfolder organ name')
    parser.add_argument('--output_path', type=Path, default=script_dir / 'final_train_list_multilabel.csv', 
                        help='Path to save final list')
    parser.add_argument('--existing_split_path', type=Path, default=None, 
                        help='Path to existing final_train_list_multilabel.csv to inherit fold assignments from (prevents Fold Drift)')
    parser.add_argument('--expected_count', type=int, default=1900, 
                        help='Expected number of valid samples for fold alignment check (default: 1900)')
    parser.add_argument('--force_resplit', action='store_true', 
                        help='Force re-running MultilabelStratifiedKFold even if existing split is available')
    parser.add_argument('--downsample_factor', type=int, default=1, help='Downsample factor used in Stage 1 (1 for 20x)')
    parser.add_argument('--n_splits', type=int, default=6, help='Number of CV splits (1 test + 5 train/val)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for stratification')
    args = parser.parse_args()

    if not args.label_path.exists():
        print(f"❌ 错误: 找不到标签文件 {args.label_path}，请先运行 Stage 2 Parse Labels。")
        return

    # 0. 检测是否已有历史分折文件 (防 Fold 漂移机制)
    existing_split_file = None
    if args.existing_split_path and args.existing_split_path.exists():
        existing_split_file = args.existing_split_path
    elif not args.force_resplit and args.output_path.exists():
        existing_split_file = args.output_path

    id_to_fold = None
    if existing_split_file is not None and not args.force_resplit:
        try:
            old_df = pd.read_csv(existing_split_file)
            id_col = 'wsi_id' if 'wsi_id' in old_df.columns else ('id' if 'id' in old_df.columns else None)
            if id_col and 'fold' in old_df.columns:
                id_to_fold = dict(zip(
                    old_df[id_col].astype(str).apply(lambda x: Path(x).stem),
                    old_df['fold'].astype(str)
                ))
                print(f"📁 检测到已有历史分折文件: {existing_split_file} (包含 {len(id_to_fold)} 条样本 Fold 映射)")
                print("   🛡️ [防漂移机制已激活]: 将优先无损继承原 Fold 划分，确保与 CONCH/Virchow2 验证集 100% 对齐！")
        except Exception as e:
            print(f"⚠️ 读取历史分折文件失败: {e}，将回退至常规分折流程。")
            id_to_fold = None

    # 1. 读取标签
    df = pd.read_csv(args.label_path)
    print(f"读取标签文件: {len(df)} 行")
    
    # 【关键检查】确认详细分级列是否存在
    # 这一步是为了防止 Stage 2 没跑对，导致重要的 grade 信息丢失
    detailed_cols = [c for c in df.columns if c.startswith('nst_grade') or c.startswith('dcis_')]
    print(f"✅ 检测到 {len(detailed_cols)} 个详细子标签列: {detailed_cols}")
    if len(detailed_cols) == 0:
        print("⚠️ 警告: 未检测到子标签 (nst_grade/dcis)。请确认是否运行了更新后的 Stage 2 脚本！")

    # 2. 检查三模型特征及坐标文件是否存在 (三模交集清洗)
    virchow_dir = Path(args.features_root) / "virchow2" / args.organ
    conch_dir = Path(args.features_root) / "conch" / args.organ
    hoptimus_dir = Path(args.features_root) / "h-optimus-1" / args.organ  # 【新增】
    # 自动兼容短横线 'h-optimus-1' 与下划线 'h_optimus_1' 物理目录名
    if not hoptimus_dir.exists() and (Path(args.features_root) / "h_optimus_1" / args.organ).exists():
        hoptimus_dir = Path(args.features_root) / "h_optimus_1" / args.organ

    valid_data = []
    print(f"正在对齐并检查三模型特征文件:")
    print(f" - Virchow2 目录: {virchow_dir}")
    print(f" - CONCH 目录: {conch_dir}")
    print(f" - H-optimus-1 目录: {hoptimus_dir}")  # 【新增】
    
    found_count = 0
    missing_count = 0
    ds = args.downsample_factor
    
    for idx, row in df.iterrows():
        wsi_id_full = str(row['id'])
        wsi_id = Path(wsi_id_full).stem  # 安全去除 .tiff / .svs 后缀
        
        # 构造预期的四个核心文件路径
        v2_feat = virchow_dir / f"{wsi_id}_features_downsampled{ds}x.npy"
        conch_feat = conch_dir / f"{wsi_id}_features_downsampled{ds}x.npy"
        hopt_feat = hoptimus_dir / f"{wsi_id}_features_downsampled{ds}x.npy"  # 【新增】
        
        # 坐标文件链式回退：优先 Virchow2 -> 其次 H-optimus-1 -> 保底 CONCH
        coord_file = virchow_dir / f"{wsi_id}_coords_downsampled{ds}x.npy"
        if not coord_file.exists():
            coord_file = hoptimus_dir / f"{wsi_id}_coords_downsampled{ds}x.npy"
        if not coord_file.exists():
            coord_file = conch_dir / f"{wsi_id}_coords_downsampled{ds}x.npy"

        # 【核心约束】只有当三模型特征和坐标完全齐备时，才纳入有效数据集（保障后续三模型融合 100% 对齐）
        if v2_feat.exists() and conch_feat.exists() and hopt_feat.exists() and coord_file.exists():
            row_dict = row.to_dict()
            row_dict['wsi_id'] = wsi_id
            row_dict['virchow2_feature_path'] = str(v2_feat)
            row_dict['conch_feature_path'] = str(conch_feat)
            row_dict['h_optimus_1_feature_path'] = str(hopt_feat)  # 【新增】与 stage3 严格对应的列名
            row_dict['coord_path'] = str(coord_file)
            
            valid_data.append(row_dict)
            found_count += 1
        else:
            missing_count += 1
            missing_desc = []
            if not v2_feat.exists(): missing_desc.append("Virchow2缺失")
            if not conch_feat.exists(): missing_desc.append("CONCH缺失")
            if not hopt_feat.exists(): missing_desc.append("H-optimus-1缺失")
            if not coord_file.exists(): missing_desc.append("Coords缺失")
            print(f"Warning: 样本 {wsi_id} 文件不完整: {', '.join(missing_desc)}")
        
    if not valid_data:
        print("❌ 错误: 没有找到任何满足三特征齐备的样本！请检查 Stage 1 是否已提取完毕。")
        return
    
    print(f"✅ 三特征交集对齐完成: 成功 {found_count} 例, 缺失或不全 {missing_count} 例")
    df_valid = pd.DataFrame(valid_data)

    # 3. 确定 Fold 划分方案 (无损继承历史 Fold vs 重新分层切分)
    inherited_success = False

    if id_to_fold is not None and not args.force_resplit:
        matched_folds = [id_to_fold.get(wsi) for wsi in df_valid['wsi_id']]
        unmatched_count = sum(1 for f in matched_folds if f is None)

        if unmatched_count == 0:
            df_valid['fold'] = matched_folds
            inherited_success = True
            print(f"\n🔒 【核心防漂移保护生效】成功 100% 继承历史 Fold 划分 ({len(df_valid)}/{len(df_valid)} 例)！")
            print("   ✅ 所有样本 Fold 与此前训练的 CONCH / Virchow2 严格对齐，零洗牌风险！")
            if len(df_valid) < len(id_to_fold):
                missing_old = set(id_to_fold.keys()) - set(df_valid['wsi_id'])
                print(f"   ⚠️ 提示: 历史表有 {len(id_to_fold)} 例，本次对齐成功 {len(df_valid)} 例 (少 {len(missing_old)} 例: {list(missing_old)[:5]}...)")
                print("      由于启用了继承模式，现有样本的 Fold 绝对锁定不变，不影响其余样本的交叉验证一致性。")
        else:
            print(f"\n⚠️ 警告: 现有有效样本中有 {unmatched_count} 例未在历史分折表中找到映射，无法无损继承！")

    if not inherited_success:
        # 【防护断言】核验样本数一致性 (防止无意间的重新洗牌)
        if args.expected_count is not None and found_count != args.expected_count:
            print(f"\n❌ 【致命风险警告: 检测到潜在 Fold 漂移 (Fold Drift)】")
            print(f"   有效样本数从预期基准 {args.expected_count} 变为 {found_count}！")
            print(f"   ⚠️ 严重后果: 样本总数改变会导致 MultilabelStratifiedKFold 全局重新洗牌！")
            print(f"   ⚠️ 重新切分会导致新生成的 Fold 0~4 与已训练的 CONCH/Virchow2 验证集彻底错位！")
            print(f"   ⚠️ Stage 4 的三模型 OOF 概率融合与集成评测将直接失效！")
            if not args.force_resplit:
                print("\n🛑 脚本已自动熔断拦截以保护现有模型：")
                print("   - 若希望无损继承旧表划分，请指定: --existing_split_path <历史CSV路径>")
                print("   - 若确认废弃旧模型并执意全局重新洗牌，请添加: --force_resplit")
                return

        label_cols = [c for c in df_valid.columns if c.startswith('label_')]
        Y_multilabel = df_valid[label_cols].fillna(0).values.astype(int)
        X_dummy = np.zeros(len(df_valid))

        mskf = MultilabelStratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        df_valid['fold'] = -1
        print(f"\n执行全新的多标签分层抽样 (n_splits={args.n_splits}, seed={args.seed})...")

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

    # 6. 保存与安全备份
    if args.output_path.exists():
        backup_path = args.output_path.with_suffix('.csv.bak')
        try:
            shutil.copy2(args.output_path, backup_path)
            print(f"📦 已安全备份原分折文件至: {backup_path}")
        except Exception as e:
            pass

    df_valid.to_csv(args.output_path, index=False)
    print(f"\n✅ 最终列表已生成: {args.output_path}")


if __name__ == "__main__":
    main()