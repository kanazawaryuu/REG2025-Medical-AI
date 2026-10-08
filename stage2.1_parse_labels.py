import json
import pandas as pd
import numpy as np
import re
import argparse 
from pathlib import Path

def parse_breast_report_multilabel(report_text):
    """
    针对乳腺癌病理报告的 22 维多标签解析器。
    剔除 <6 例伪任务，吸收极罕见亚型，新增 FCC 与大汗腺化生。
    """
    # 1. 初始化 22 维金标准标签字典 (全员样本数 >= 6)
    labels = {
        # --- 恶性浸润癌与原位癌 (8类) ---
        "label_invasive_nst": 0,          # 非特殊型浸润癌 [共 924 例: NST 915例(G2 630+G1 169+G3 116) + 化生性癌 4例 + 浸润性筛状癌 2例 + 管状癌 2例 + 伴大汗腺分化癌 1例]
        "label_dcis": 0,                  # 导管原位癌 [共 453 例: 典型DCIS 451例(单纯284+伴浸润癌153+伴乳头状瘤2) + 实性乳头状原位癌 2例]
        "label_invasive_lobular": 0,      # 浸润性小叶癌 [共 78 例: 经典型 77例 + 多形性亚型 1例; 其中14例合并LCIS]
        "label_mucinous": 0,              # 黏液癌 [共 53 例: 经典黏液癌 43例 + 伴黏液癌特征浸润癌 10例]
        "label_lcis": 0,                  # 小叶原位癌/小叶肿瘤形成 [共 22 例: LCIS 21例(单纯3+伴ILC 14+伴乳头状瘤2) + 非典型小叶增生ALH 1例]
        "label_micropapillary": 0,        # 浸润性微乳头状癌 [共 9 例: 独立/主病变 7例 + 合并DCIS 2例]
        "label_lymphoma": 0,              # 恶性淋巴瘤 [共 6 例: 均为原发或累及乳腺的恶性淋巴瘤独立病变]
        "label_micro_invasive": 0,        # 微浸润癌 [共 6 例: 浸润灶最大径≤1mm，6例全部合并广泛DCIS]
        
        # --- 纤维上皮与乳头状谱系 (5类) ---
        "label_papillary_neoplasm": 0,    # 乳头状肿瘤大类 [共 204 例: 导管内乳头状瘤/乳头状肿瘤 202例(独立183+伴UDH 21+伴ADH 2+伴DCIS 2) + 实性乳头状原位癌 2例]
        "label_fibroepithelial": 0,       # 纤维上皮性及间质肿瘤大类 [共 139 例: 纤维上皮性肿瘤NOS 55例 + 纤维腺瘤FA 54例 + 纤维腺瘤样改变 15例 + 叶状肿瘤PT 11例 + 韧带样纤维瘤病 4例]
        "label_fibroadenoma": 0,          # 纤维腺瘤特指 [共 69 例: 确诊FA 23例 + 倾向FA(favor FA) 31例 + 纤维腺瘤样改变 15例]
        "label_papilloma": 0,             # 导管内乳头状瘤特指 [共 41 例: 伴UDH 20例 + 单纯乳头状瘤 18例 + 伴DCIS 2例 + 伴大汗腺化生 1例]
        "label_phyllodes": 0,             # 叶状肿瘤特指 [共 11 例: 确诊PT 4例 + 倾向叶状肿瘤(favor PT) 7例]
        
        # --- 增生、腺病与良性瘤样病变 (6类) ---
        "label_udh": 0,                   # 普通型导管增生 [共 58 例: 独立存在 37例 + 合并于乳头状肿瘤 21例]
        "label_columnar": 0,              # 柱状细胞病变 [共 50 例: 柱状细胞病变CCL 46例 + 柱状细胞改变CCC 3例 + 柱状细胞增生CCH 1例]
        "label_adh": 0,                   # 非典型导管增生 [共 36 例: 独立/伴良性病变 34例 + 合并乳头状肿瘤 2例]
        "label_fibrocystic": 0,           # 纤维囊性改变/良性瘤样改变 [共 53 例: 纤维囊性改变FCC 35例 + 导管扩张 17例 + 黏液囊肿样病变 1例]
        "label_sclerosing_adenosis": 0,   # 硬化性腺病/腺病谱系 [共 28 例: 硬化性腺病 25例(独立13+伴微钙化12) + 大汗腺腺病 3例]
        "label_fea": 0,                   # 平坦上皮不典型增生 [共 6 例: 独立/主病变 4例 + 伴随病变 2例]
        
        # --- 伴随征象与阴性对照 (3类) ---
        "label_microcalcification": 0,    # 微钙化征象 [共 162 例: 伴DCIS 68例 + 伴NST 36例 + 伴硬化性腺病 12例 + 伴其他良恶性病变 46例]
        "label_no_tumor": 0,              # 未见肿瘤证据/反应性改变 [共 47 例: 未见肿瘤证据 34例 + 异物反应 6例 + 肉芽肿性小叶炎 3例 + 脂肪坏死 2例 + 放疗后异型 2例]
        "label_apocrine_metaplasia": 0    # 大汗腺化生征象 [共 27 例: 独立/主要 5例 + 伴发于柱状病变/囊性变/癌 22例]
    }

    # 空值防护
    if pd.isna(report_text) or not isinstance(report_text, str):
        return labels

    text = report_text.lower()

    # 1. 明确的特定浸润癌亚型 (采用严格实体/短语匹配，消除跨句串扰)
    if re.search(r'\bmucinous\s+carcinoma\b|carcinoma\s+with\s+(?:features\s+of\s+)?mucinous\b', text):
        labels["label_mucinous"] = 1

    # 严格绑定微乳头状与浸润实体，杜绝伴发 DCIS micropapillary 时的跨句误判
    if re.search(r'\b(?:invasive|infiltrating)\s+micro[\s\-]?papillary\b|\bmicro[\s\-]?papillary\s+(?:invasive|infiltrating)\s+carcinoma\b', text):
        labels["label_micropapillary"] = 1

    # 严格绑定小叶与浸润实体，彻底杜绝 NST 合并 LCIS 或小叶炎时的跨句误判
    if re.search(r'\b(?:invasive|infiltrating)\s+lobular\b|\blobular\s+(?:invasive|infiltrating)\b|\bilc\b', text):
        labels["label_invasive_lobular"] = 1

    is_micro = bool(re.search(r'\bmicro[\s\-]?invasive\b', text))
    if is_micro:
        labels["label_micro_invasive"] = 1

    # 2. 非特殊型浸润癌 (NST) 及吸收极罕见亚型 (管状2例/筛状2例/化生4例/大汗腺癌1例)
    is_other_specific_invasive = (labels["label_mucinous"] or labels["label_micropapillary"] or labels["label_invasive_lobular"])
    
    # 精准定义吸收亚型
    has_absorbed_rare_invasive = (
        "tubular carcinoma" in text or 
        bool(re.search(r'\b(?:invasive|infiltrating)\s+(?:carcinoma\s*,?\s*)?cribriform\b|\bcribriform\s+(?:invasive|infiltrating)\b', text)) or 
        "metaplastic carcinoma" in text or 
        "apocrine differentiation" in text or
        "apocrine carcinoma" in text
    )

    # 兼容 invasive ductal carcinoma (IDC) / infiltrating 等写法
    has_invasive_generic = (("invasive" in text or "infiltrating" in text) and "carcinoma" in text) or bool(re.search(r'\b(nst|idc)\b', text))

    if "no special type" in text or re.search(r'\b(nst|idc)\b', text):
        labels["label_invasive_nst"] = 1
    elif has_invasive_generic and not is_other_specific_invasive and not is_micro:
        labels["label_invasive_nst"] = 1
    elif has_absorbed_rare_invasive:
        labels["label_invasive_nst"] = 1

    # 3. 原位癌 (吸收实性乳头状原位癌 2例 与 ALH 1例)
    if "ductal carcinoma in situ" in text or re.search(r'\bdcis\b', text):
        labels["label_dcis"] = 1
    if "solid papillary" in text and ("in situ" in text or "carcinoma" in text):
        labels["label_dcis"] = 1
        labels["label_papillary_neoplasm"] = 1
        
    if "lobular carcinoma in situ" in text or re.search(r'\blcis\b', text) or "atypical lobular hyperplasia" in text or re.search(r'\balh\b', text):
        labels["label_lcis"] = 1

    # 4. 纤维上皮与间质 (吸收韧带样纤维瘤病 4例)
    if "fibroepithelial" in text:
        labels["label_fibroepithelial"] = 1
    if "phyllodes" in text:
        labels["label_phyllodes"] = 1
        labels["label_fibroepithelial"] = 1
    if "fibroadenoma" in text:
        labels["label_fibroadenoma"] = 1
        labels["label_fibroepithelial"] = 1
    if "fibromatosis" in text or "desmoid" in text:
        labels["label_fibroepithelial"] = 1

    # 5. 乳头状病变 (排除 micro-papillary / micro papillary 干扰)
    if re.search(r'(?<!micro)(?<!micro\s)(?<!micro\-)\bpapillary\b', text) and ("neoplasm" in text or "lesion" in text or "tumor" in text or "carcinoma" in text):
        labels["label_papillary_neoplasm"] = 1
    if "papilloma" in text:
        labels["label_papilloma"] = 1
        labels["label_papillary_neoplasm"] = 1

    # 6. 增生与癌前病变 (使用正则单词边界防标点拦截)
    if "atypical ductal hyperplasia" in text or re.search(r'\badh\b', text):
        labels["label_adh"] = 1
    if "flat epithelial atypia" in text or re.search(r'\bfea\b', text):
        labels["label_fea"] = 1
    if "usual ductal hyperplasia" in text or re.search(r'\budh\b', text):
        labels["label_udh"] = 1
    if "columnar cell" in text:
        labels["label_columnar"] = 1
    if "sclerosing adenosis" in text or "apocrine adenosis" in text:
        labels["label_sclerosing_adenosis"] = 1

    # 7. 瘤样改变、征象与对照
    if "fibrocystic" in text or "duct ectasia" in text or "mucocele" in text:
        labels["label_fibrocystic"] = 1
    if "apocrine metaplasia" in text or "apocrine change" in text:
        labels["label_apocrine_metaplasia"] = 1
    if "microcalcification" in text:
        labels["label_microcalcification"] = 1
    if "lymphoma" in text:
        labels["label_lymphoma"] = 1

    # 8. 排除性/良性阴性兜底 (排除微钙化与大汗腺化生征象干扰)
    non_disease_keys = {"label_no_tumor", "label_microcalcification", "label_apocrine_metaplasia"}
    has_positive_disease = any(v == 1 for k, v in labels.items() if k not in non_disease_keys)

    benign_terms = [
        "no evidence", "no tumor", "inflammation", "mastitis", 
        "foreign body", "fat necrosis", "radiation", "pseudoangiomatous", "pash"
    ]
    if not has_positive_disease and any(term in text for term in benign_terms):
        labels["label_no_tumor"] = 1

    return labels

def extract_detailed_labels(report_text):
    """
    提取 NST Nottingham 分级与 DCIS 详细特征。
    自动推导 nst_grade_overall (1/2/3 级)。
    """
    data = {
        'nst_grade_tubule': -1,
        'nst_grade_nuclear': -1,
        'nst_grade_mitoses': -1,
        'nst_grade_overall': -1,  # [新增] 1=Grade I, 2=Grade II, 3=Grade III (-1 为缺失)
        'dcis_grade': -1,
        'dcis_necrosis': -1,
        'dcis_type_solid': 0,          
        'dcis_type_cribriform': 0,     
        'dcis_type_micropapillary': 0, 
    }
    
    if pd.isna(report_text) or not isinstance(report_text, str):
        return data
        
    text = report_text.lower()
    
    # 1. 提取 Nottingham 各单项得分 (仅在明确存在浸润癌或腺管评分时生效)
    is_invasive_case = ("invasive" in text or "infiltrating" in text or "nst" in text or "idc" in text)
    
    t_match = re.search(r'tubule(?:s|\s+formation)?:\s*(\d)', text)
    n_match = re.search(r'nuclear grade:\s*(\d)', text)
    m_match = re.search(r'mitoses:\s*(\d)', text)
    
    if t_match: data['nst_grade_tubule'] = int(t_match.group(1))
    if m_match: data['nst_grade_mitoses'] = int(m_match.group(1))
    
    # 只有当样本属于浸润癌且非纯 DCIS 语境下，才将数字核分级赋给 NST
    if n_match and is_invasive_case:
        data['nst_grade_nuclear'] = int(n_match.group(1))
    
    # 自动推导 Nottingham 临床总分 (总分 3-9 分映射至 I~III 级)
    if data['nst_grade_tubule'] > 0 and data['nst_grade_nuclear'] > 0 and data['nst_grade_mitoses'] > 0:
        total = data['nst_grade_tubule'] + data['nst_grade_nuclear'] + data['nst_grade_mitoses']
        if 3 <= total <= 5:
            data['nst_grade_overall'] = 1
        elif 6 <= total <= 7:
            data['nst_grade_overall'] = 2
        elif 8 <= total <= 9:
            data['nst_grade_overall'] = 3
    else:
        # 若未提供单项细分，直接匹配报告中的总评级 (如 Nottingham grade 2 / Grade III)
        direct_match = re.search(r'(?:nottingham|histologic(?:al)?)\s+grade(?:\s*[:\-]?\s*|\s+)(?:grade\s*)?([123]|i{1,3})\b', text)
        if direct_match:
            val_str = direct_match.group(1)
            roman_map = {'i': 1, 'ii': 2, 'iii': 3, '1': 1, '2': 2, '3': 3}
            data['nst_grade_overall'] = roman_map.get(val_str, -1)
        
        # 原有匹配未命中时，在浸润癌语境下放宽匹配普通 Grade 1~3 / I~III (排除前面的 nuclear)
        if data['nst_grade_overall'] == -1 and is_invasive_case:
            m_fallback = re.search(r'(?<!nuclear\s)\bgrade(?:\s*[:\-]?\s*|\s+)([123]|i{1,3})\b', text)
            if m_fallback:
                roman_map = {'i': 1, 'ii': 2, 'iii': 3, '1': 1, '2': 2, '3': 3}
                data['nst_grade_overall'] = roman_map.get(m_fallback.group(1), -1)

    # 2. 提取 DCIS 亚型特征 (在 DCIS 专属切片内匹配，彻底杜绝浸润癌核分级串扰)
    if 'ductal carcinoma in situ' in text or 'dcis' in text:
        # 截取 DCIS 关键词之后的子文本
        dcis_sections = re.split(r'\b(?:ductal carcinoma in situ|dcis)\b', text)
        dcis_text = dcis_sections[-1] if len(dcis_sections) > 1 else text
        
        # A. 核分级 (仅严格匹配 DCIS 临床文本定义)
        if re.search(r'nuclear grade:\s*high\b', dcis_text): 
            data['dcis_grade'] = 3
        elif re.search(r'nuclear grade:\s*intermediate\b', dcis_text): 
            data['dcis_grade'] = 2
        elif re.search(r'nuclear grade:\s*low\b', dcis_text): 
            data['dcis_grade'] = 1
            
        # B. 坏死状态 (仅在 DCIS 切片内检索)
        if 'comedo' in dcis_text: 
            data['dcis_necrosis'] = 2
        elif 'necrosis: present' in dcis_text: 
            data['dcis_necrosis'] = 1
        elif 'necrosis: absent' in dcis_text: 
            data['dcis_necrosis'] = 0
            
        # C. 结构类型
        dcis_type_matches = re.finditer(r'(?<!histologic\s)(?<!histological\s)type:\s*([^\r\n\.;]+)', dcis_text)
        for match in dcis_type_matches:
            type_section = match.group(1).lower()
            if 'solid' in type_section: data['dcis_type_solid'] = 1
            if 'cribriform' in type_section: data['dcis_type_cribriform'] = 1
            if 'micropapillary' in type_section: data['dcis_type_micropapillary'] = 1

    return data


# --- 执行主程序 (改动后：彻底告别冗长参数，支持直接运行) ---

def main():
    # 自动获取脚本所在的绝对路径目录
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Parse Breast Cancer Pathology Reports")
    # 默认路径自动绑定到脚本同级目录，无需手动输入
    parser.add_argument('--json_path', type=Path, default=script_dir / 'train.json', 
                        help='Path to raw train.json')
    parser.add_argument('--output_path', type=Path, default=script_dir / 'breast_cancer_multilabel_targets.csv', 
                        help='Path to save output CSV')
    args = parser.parse_args()

    print(f"正在读取 {args.json_path} ...")
    if not args.json_path.exists():
        print(f"❌ 错误: 找不到文件 {args.json_path}")
        return

    # 转换为DataFrame
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # 1. 筛选乳腺数据 (Breast 或 Nipple)
    df['organ_raw'] = df['report'].apply(lambda x: x.split(',')[0].strip() if x else "")
    breast_df = df[df['organ_raw'].isin(['Breast', 'Nipple'])].copy()
    
    print(f"筛选出乳腺/乳头样本: {len(breast_df)} 例")
    
    # 2. 解析基础 22 维金标准多标签
    print("Step 1: 解析基础 22 维金标准多标签...")
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
    label_cols = [c for c in final_df.columns if c.startswith('label_')]
    print(final_df[label_cols].sum().sort_values(ascending=False))
    print("\n--- NST Grade 子标签统计 ---")
    nst_cols = [c for c in final_df.columns if c.startswith('nst_grade')]
    print(final_df[nst_cols].apply(pd.Series.value_counts).fillna(0))
    print("\n--- DCIS 子标签统计 ---")
    dcis_cols = [c for c in final_df.columns if c.startswith('dcis_')]
    print(final_df[dcis_cols].apply(pd.Series.value_counts).fillna(0))
    
    # 6. 保存
    final_df.to_csv(args.output_path, index=False)
    print(f"\n✅ 处理完成！已保存至: {args.output_path}")
    print(f"总列数: {len(final_df.columns)} (含 ID, Report, 22个大类, 9个子特征)")

if __name__ == "__main__":
    main()