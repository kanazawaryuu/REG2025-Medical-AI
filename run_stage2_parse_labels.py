import json
import pandas as pd
import numpy as np
import re
import argparse 

def parse_breast_report_multilabel(report_text):
    """
    针对乳腺癌病理报告的多标签解析器。
    返回一个字典，包含所有关键病理类型的二值标签 (0/1)。
    """
    text = report_text.lower()
    
    # 初始化标签字典 (Schema)
    # 这些列是根据您的数据集统计定制的
    labels = {
        # --- 1. 浸润性癌亚型 (Invasive Subtypes) ---
        "label_invasive_nst": 0,          # 非特殊类型浸润性癌 (IDC/NST)            [约 915 例: Grade II 630 + Grade I 169 + Grade III 116]
        "label_invasive_lobular": 0,      # 浸润性小叶癌 (ILC)                      [约 78 例: 77(Invasive lobular carcinoma) + 1(Invasive lobular carcinoma, pleomorphic type)]
        "label_mucinous": 0,              # 黏液癌 (纯型或混合型)                   [约 53 例: 43(Mucinous carcinoma) + 10(Invasive carcinoma with features of mucinous carcinoma)]
        "label_micropapillary": 0,        # 浸润性微乳头状癌                        [约 8 例: 8(Invasive micropapillary carcinoma)]
        "label_metaplastic": 0,           # 化生性癌                                [约 4 例: 4(Metaplastic carcinoma)]
        "label_micro_invasive": 0,        # 微浸润癌                                [约 6 例: 6(Micro-invasive carcinoma)]
        "label_cribriform_invasive": 0,   # 浸润性筛状癌                            [约 2 例: 2(Invasive cribriform carcinoma)]
        "label_tubular_carcinoma": 0,     # 小管癌                                  [约 2 例: 2(Tubular carcinoma)]
        
        # --- 2. 原位癌 (In Situ) ---
        "label_dcis": 0,                  # 导管原位癌                              [约 451 例: 449(Ductal carcinoma in situ) + 2(Ductal carcinoma in situ in intraductal papilloma)]
        "label_lcis": 0,                  # 小叶原位癌                              [约 21 例: 21(Lobular carcinoma in situ)]
        
        # --- 3. 良性/交界性肿瘤 (Benign/Borderline Tumors) ---
        "label_fibroepithelial": 0,       # 纤维上皮肿瘤 (包含纤维腺瘤/叶状肿瘤)    [约 135 例: 55(Fibroepithelial tumor) + 30(Fibroepithelial tumor, favor fibroadenoma) + 23(Fibroadenoma) + 15(Fibroadenomatoid change) + 7(Fibroepithelial tumor, favor phyllodes tumor) + 4(Phyllodes tumor) + 1(Fibroepithelial lesion, favor fibroadenoma)]
        "label_phyllodes": 0,             # 叶状肿瘤 (特指)                         [约 11 例: 7(Fibroepithelial tumor, favor phyllodes tumor) + 4(Phyllodes tumor)]
        "label_fibroadenoma": 0,          # 纤维腺瘤 (特指)                         [约 69 例: 30(Fibroepithelial tumor, favor fibroadenoma) + 23(Fibroadenoma) + 15(Fibroadenomatoid change) + 1(Fibroepithelial lesion, favor fibroadenoma)]
        "label_papillary_neoplasm": 0,    # 乳头状肿瘤 (包含导管内乳头状瘤)         [约 200 例: 155(Papillary neoplasm) + 20(Intraductal papilloma with UDH) + 18(Intraductal papilloma) + 其他少量混合型]
        "label_papilloma": 0,             # 导管内乳头状瘤 (特指)                   [约 41 例: 20(Intraductal papilloma with UDH) + 18(Intraductal papilloma) + 1(Intraductal papilloma with apocrine metaplasia) + (Ductal carcinoma in situ in intraductal papilloma)]
        
        # --- 4. 增生与癌前病变 (Proliferative/Precursor) ---
        "label_adh": 0,                   # 非典型导管增生 (ADH)                    [约 36 例: 34(Atypical ductal hyperplasia) + 2(Papillary neoplasm with ADH)]
        "label_alh": 0,                   # 非典型小叶增生 (ALH)                    [约 1 例: 1(Atypical lobular hyperplasia)]
        "label_fea": 0,                   # 平坦型上皮非典型增生 (FEA)              [约 6 例: 6(Flat epithelial atypia)]
        "label_udh": 0,                   # 普通型导管增生 (UDH)                    [约 58 例: 37(Usual ductal hyperplasia) + 20(Intraductal papilloma with UDH) + 1(Papillary neoplasm with UDH)]
        "label_columnar": 0,              # 柱状细胞病变 (CCL/CCC/CCH)              [约 50 例: 46(Columnar cell lesion) + 3(Columnar cell change) + 1(Columnar cell hyperplasia)]
        "label_sclerosing_adenosis": 0,   # 硬化性腺病                              [约 25 例: 25(Sclerosing adenosis)]
        
        # --- 5. 其他/淋巴瘤 ---
        "label_lymphoma": 0,              # 淋巴瘤                                  [约 6 例: 6(Malignant lymphoma)]
        "label_microcalcification": 0,    # 微钙化                                  [约 161+ 例: 161(Microcalcification) + 大量作为次要诊断出现的病例]
        "label_no_tumor": 0               # 未见肿瘤/正常/炎症                      [约 100+ 例: 34(No evidence of tumor) + 35(Fibrocystic change) + 25(Apocrine metaplasia) + 17(Duct ectasia) 等非肿瘤病变]
    }

    # --- 解析逻辑 (正则匹配) ---

    # 1. 黏液癌 (Mucinous)
    # 注意：排除 "mucocele" (黏液囊肿)
    if "mucinous" in text and "carcinoma" in text:
        labels["label_mucinous"] = 1

    # 2. 浸润性微乳头状癌 (Micropapillary)
    if "micropapillary" in text and "invasive" in text:
        labels["label_micropapillary"] = 1

    # 3. 化生性癌 (Metaplastic)
    if "metaplastic" in text:
        labels["label_metaplastic"] = 1
        
    # 4. 浸润性小叶癌 (Lobular Invasive)
    if "lobular" in text and "invasive" in text:
        labels["label_invasive_lobular"] = 1
        # 某些混合型可能同时包含 NST 和 Lobular，但在竞赛中通常优先标记特定亚型

    # 5. 微浸润 (Micro-invasive)
    if "micro-invasive" in text or "microinvasive" in text:
        labels["label_micro_invasive"] = 1

    # 6. 浸润性筛状癌 (Invasive Cribriform)
    if "cribriform" in text and "invasive" in text:
        labels["label_cribriform_invasive"] = 1

    # 7. 小管癌 (Tubular)
    if "tubular carcinoma" in text:
        labels["label_tubular_carcinoma"] = 1

    # 8. 浸润性癌 (NST / Generic)
    # 只有当它不是上述特殊亚型时，或者报告明确写了 "no special type" 时才标记
    # 如果报告只写了 "Invasive carcinoma" 且没提其他特征，也归为此类
    is_special_type = (labels["label_mucinous"] or labels["label_micropapillary"] or 
                       labels["label_metaplastic"] or labels["label_invasive_lobular"] or
                       labels["label_tubular_carcinoma"] or labels["label_cribriform_invasive"])
    
    if "no special type" in text:
        labels["label_invasive_nst"] = 1
    elif "invasive carcinoma" in text and not is_special_type and "micro-invasive" not in text:
        # 这是一个兜底逻辑，如果没被标记为特殊亚型，则归为 NST
        labels["label_invasive_nst"] = 1

    # --- 原位癌 ---
    if "ductal carcinoma in situ" in text or "dcis" in text:
        labels["label_dcis"] = 1
    if "lobular carcinoma in situ" in text or "lcis" in text:
        labels["label_lcis"] = 1
        
    # --- 良性/交界性 ---
    if "fibroepithelial" in text:
        labels["label_fibroepithelial"] = 1 # 这是一个大类
    if "phyllodes" in text:
        labels["label_phyllodes"] = 1
        labels["label_fibroepithelial"] = 1 # 叶状肿瘤属于纤维上皮肿瘤
    if "fibroadenoma" in text: # 包含 fibroadenomatoid
        labels["label_fibroadenoma"] = 1
        labels["label_fibroepithelial"] = 1 # 纤维腺瘤属于纤维上皮肿瘤
        
    if "papillary" in text and ("neoplasm" in text or "lesion" in text or "tumor" in text or "carcinoma" in text):
        labels["label_papillary_neoplasm"] = 1
    if "papilloma" in text:
        labels["label_papilloma"] = 1
        labels["label_papillary_neoplasm"] = 1
        
    # --- 增生/癌前 ---
    if "atypical ductal hyperplasia" in text or " adh " in text:
        labels["label_adh"] = 1
    if "atypical lobular hyperplasia" in text or " alh " in text:
        labels["label_alh"] = 1
    if "flat epithelial atypia" in text or " fea " in text:
        labels["label_fea"] = 1
    if "usual ductal hyperplasia" in text or " udh " in text:
        labels["label_udh"] = 1
    if "columnar cell" in text:
        labels["label_columnar"] = 1
    if "sclerosing adenosis" in text:
        labels["label_sclerosing_adenosis"] = 1
        
    # --- 其他 ---
    if "lymphoma" in text:
        labels["label_lymphoma"] = 1
    if "microcalcification" in text:
        labels["label_microcalcification"] = 1
        
    # --- 排除/阴性 ---
    # 只有当没有检测到任何肿瘤/增生关键词，且报告包含 "no evidence" 或 "no tumor" 时
    has_positive_finding = any(v == 1 for k, v in labels.items() if k != "label_no_tumor")
    if not has_positive_finding and ("no evidence" in text or "no tumor" in text or "inflammation" in text or "mastitis" in text):
        labels["label_no_tumor"] = 1
        
    return labels

def extract_detailed_labels(report_text):
    """
    【新增】从报告中提取 NST 分级和 DCIS 详细特征
    策略：基于统计数据，仅保留样本数 > 10 的高频特征，且支持 DCIS 多种类型共存
    """
    data = {
        # --- NST Grading (Nottingham Score) ---
        # 范围: 1, 2, 3 (-1 表示未提及)
        'nst_grade_tubule': -1,
        'nst_grade_nuclear': -1,
        'nst_grade_mitoses': -1,
        
        # --- DCIS Nuclear Grade (核分级) ---
        # 1=Low, 2=Intermediate, 3=High (-1 表示未提及)
        'dcis_grade': -1,
        
        # --- DCIS Necrosis (坏死) ---
        # 0=Absent, 1=Present(Focal), 2=Present(Comedo-type) (-1 表示未提及)
        'dcis_necrosis': -1,
        
        # --- DCIS Architecture Types (结构类型 - 多标签) ---
        # 仅保留样本数 > 10 的类型: Solid(160), Cribriform(85), Micropapillary(12)
        'dcis_type_solid': 0,          
        'dcis_type_cribriform': 0,     
        'dcis_type_micropapillary': 0, 
    }
    
    if pd.isna(report_text) or not isinstance(report_text, str):
        return data
        
    text = report_text.lower()
    
    # === 1. 提取 NST 分级 (仅针对 Invasive Carcinoma of NST) ===
    if 'invasive carcinoma of no special type' in text:
        # 匹配标准格式: Tubule formation: 3, Nuclear grade: 2, Mitoses: 1
        t_match = re.search(r'tubule formation:\s*(\d)', text)
        n_match = re.search(r'nuclear grade:\s*(\d)', text) # NST 的核分级
        m_match = re.search(r'mitoses:\s*(\d)', text)
        
        if t_match: data['nst_grade_tubule'] = int(t_match.group(1))
        if n_match: data['nst_grade_nuclear'] = int(n_match.group(1))
        if m_match: data['nst_grade_mitoses'] = int(m_match.group(1))

    # === 2. 提取 DCIS 详细特征 (仅针对 DCIS) ===
    if 'ductal carcinoma in situ' in text:
        
        # A. DCIS 核分级 (Nuclear Grade)
        # 统计: High(111), Intermediate(138), Low(31)
        if 'nuclear grade: high' in text: 
            data['dcis_grade'] = 3
        elif 'nuclear grade: intermediate' in text: 
            data['dcis_grade'] = 2
        elif 'nuclear grade: low' in text: 
            data['dcis_grade'] = 1
            
        # B. DCIS 坏死 (Necrosis)
        # 统计: Absent(144), Comedo(74), Focal(62)
        # Comedo (粉刺样) 是高危特征，单独列为 2，其他 Present 为 1
        if 'comedo' in text: 
            data['dcis_necrosis'] = 2
        elif 'necrosis: present' in text: 
            data['dcis_necrosis'] = 1
        elif 'necrosis: absent' in text: 
            data['dcis_necrosis'] = 0
            
        # C. DCIS 结构类型 (Type) - 支持多标签 (如 Cribriform and solid)
        if 'type:' in text:
            # 获取 Type 这一行的内容并转小写
            type_section = text.split('type:')[1].split('\n')[0].lower()
            
            # 使用独立的 if 语句，确保互不排斥
            if 'solid' in type_section: 
                data['dcis_type_solid'] = 1
            if 'cribriform' in type_section: 
                data['dcis_type_cribriform'] = 1
            if 'micropapillary' in type_section: 
                data['dcis_type_micropapillary'] = 1

    return data

def process_dataset(json_path, output_csv_path):
    print(f"正在读取 {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 1. 筛选乳腺数据 (Breast 或 Nipple)
    # 假设报告以 "Breast" 或 "Nipple" 开头
    df['organ_raw'] = df['report'].apply(lambda x: x.split(',')[0].strip())
    breast_df = df[df['organ_raw'].isin(['Breast', 'Nipple'])].copy()
    
    print(f"筛选出乳腺/乳头样本: {len(breast_df)} 例")
    
    # 2. 应用多标签解析
    print("正在解析病理报告标签...")
    label_list = []
    for idx, row in breast_df.iterrows():
        labels = parse_breast_report_multilabel(row['report'])
        # 将 ID 也放进去方便合并
        labels['id'] = row['id']
        labels['report_text'] = row['report'] # 保留原始报告方便核对
        label_list.append(labels)
        
    label_df = pd.DataFrame(label_list)
    
    # 将 ID 设为第一列
    cols = ['id', 'report_text'] + [c for c in label_df.columns if c not in ['id', 'report_text']]
    label_df = label_df[cols]
    
    # 3. 统计各类别数量 (验证长尾分布是否被捕获)
    print("\n--- 标签分布统计 (Multi-label) ---")
    label_cols = [c for c in label_df.columns if c.startswith('label_')]
    stats = label_df[label_cols].sum().sort_values(ascending=False)
    print(stats)
    
    # 4. 保存结果
    label_df.to_csv(output_csv_path, index=False)
    print(f"\n已保存处理后的标签文件至: {output_csv_path}")
    
    return label_df

# --- 执行主程序 ---

def main():
    parser = argparse.ArgumentParser(description="Parse Breast Cancer Pathology Reports")
    parser.add_argument('--json_path', type=str, default='train.json', help='Path to raw train.json')
    parser.add_argument('--output_path', type=str, default='breast_cancer_multilabel_targets.csv', help='Path to save output CSV')
    args = parser.parse_args()

    print(f"正在读取 {args.json_path} ...")
    try:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {args.json_path}")
        return

    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 1. 筛选乳腺数据 (Breast 或 Nipple)
    df['organ_raw'] = df['report'].apply(lambda x: x.split(',')[0].strip() if x else "")
    breast_df = df[df['organ_raw'].isin(['Breast', 'Nipple'])].copy()
    
    print(f"筛选出乳腺/乳头样本: {len(breast_df)} 例")
    
    # 2. 解析基础 24 分类标签
    print("Step 1: 解析基础 24 分类标签...")
    basic_labels_list = []
    for _, row in breast_df.iterrows():
        labels = parse_breast_report_multilabel(row['report'])
        basic_labels_list.append(labels)
    basic_labels_df = pd.DataFrame(basic_labels_list)
    
    # 3. 解析详细子特征 (NST Grade & DCIS Details)
    print("Step 2: 解析 NST 及 DCIS 详细子标签 (过滤 <10 样本)...")
    detailed_features_list = []
    for report in breast_df['report']:
        detailed_features_list.append(extract_detailed_labels(report))
    detailed_df = pd.DataFrame(detailed_features_list)
    
    # 4. 合并所有数据
    # 结构: [id, report_text] + [24个基础分类] + [8个详细子特征]
    # 重置索引以确保对齐
    breast_df.reset_index(drop=True, inplace=True)
    basic_labels_df.reset_index(drop=True, inplace=True)
    detailed_df.reset_index(drop=True, inplace=True)
    
    final_df = pd.concat([
        breast_df[['id', 'report']].rename(columns={'report': 'report_text'}), 
        basic_labels_df, 
        detailed_df
    ], axis=1)
    
    # 5. 打印统计信息
    print("\n--- 标签分布统计 ---")
    # 统计基础标签
    label_cols = [c for c in final_df.columns if c.startswith('label_')]
    print(final_df[label_cols].sum().sort_values(ascending=False).head(10))
    print("\n--- DCIS 子标签统计 ---")
    dcis_cols = [c for c in final_df.columns if c.startswith('dcis_')]
    print(final_df[dcis_cols].apply(pd.Series.value_counts).fillna(0))
    
    # 6. 保存
    final_df.to_csv(args.output_path, index=False)
    print(f"\n✅ 处理完成！已保存至: {args.output_path}")
    print(f"总列数: {len(final_df.columns)} (含 ID, Report, 24个大类, 8个子特征)")

if __name__ == "__main__":
    main()