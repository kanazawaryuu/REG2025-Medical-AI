import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import spacy
import warnings
from collections import Counter
import os

# 忽略警告
warnings.filterwarnings("ignore")

# --- 1. 配置区域 (Configuration) ---
GT_PATH = 'train.json'                      # 原始真值文件路径
PRED_PATH = 'submission.json'               # 预测结果文件路径
EMBEDDING_MODEL = 'dmis-lab/biobert-v1.1'   # BioBERT 模型
SPACY_MODEL = 'en_core_sci_lg'              # SciSpacy 模型

# 输出文件配置
OUTPUT_DETAILS = 'official_eval_details.json' 
OUTPUT_SUMMARY = 'official_eval_summary.txt'  

# --- 2. 核心评估逻辑 ---

class EmbeddingEvaluator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    @torch.no_grad()
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    def get_score(self, ref_text, hyp_text, scale=.5):
        ref_embedding = self.get_embedding(ref_text)
        hyp_embedding = self.get_embedding(hyp_text)
        if ref_embedding.ndim == 0: ref_embedding = ref_embedding.reshape(1, -1)
        if hyp_embedding.ndim == 0: hyp_embedding = hyp_embedding.reshape(1, -1)
        score = cosine_similarity([ref_embedding], [hyp_embedding])[0][0]
        if (scale != 0) and (score > scale):
            score = (score - scale) / (1 - scale)
        elif scale != 0:
            score = 0
        return score

class KeywordEvaluator:
    def __init__(self, model_name='en_core_sci_lg'):
        print(f"正在加载 SciSpacy 模型: {model_name} ...")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise OSError(f"缺失模型 '{model_name}'. 请先安装 scispacy 和该模型。")

    def get_keywords(self, text: str, min_length: int = 3):
        doc = self.nlp(text)
        keywords = []
        for ent in doc.ents:
            if len(ent.text) >= min_length:  
                keywords.append(ent.text.lower())
        return list(set(keywords))  
    
    def get_score_detailed(self, ref_text: str, hyp_text: str, min_length: int = 3):
        ref_keywords = self.get_keywords(ref_text, min_length)
        hyp_keywords = self.get_keywords(hyp_text, min_length)
        
        set_ref = set(ref_keywords)
        set_hyp = set(hyp_keywords)
        
        # Jaccard
        intersection = len(set_ref.intersection(set_hyp))
        union = len(set_ref.union(set_hyp))
        score = intersection / union if union != 0 else 0
        
        missed = list(set_ref - set_hyp)       # 漏报
        hallucinated = list(set_hyp - set_ref) # 多报
        
        return score, missed, hallucinated, ref_keywords

class REG_Evaluator_Pro:
    def __init__(self, embedding_model, spacy_model): 
        self.embedding_eval = EmbeddingEvaluator(embedding_model)
        self.key_eval = KeywordEvaluator(spacy_model)
    
    @staticmethod
    def get_bleu4(ref_text, hyp_text):
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        if len(ref_words) < 4 or len(hyp_words) < 4: return 0.0
        ref_fourgrams = [' '.join(ref_words[i:i+4]) for i in range(len(ref_words)-3)]
        hyp_fourgrams = [' '.join(hyp_words[i:i+4]) for i in range(len(hyp_words)-3)]
        count = 0
        total = 0
        for fourgram in hyp_fourgrams:
            count += min(hyp_fourgrams.count(fourgram), ref_fourgrams.count(fourgram))
            total += 1
        return count / total if total > 0 else 0.0

    @staticmethod
    def get_rouge(ref_text, hyp_text):
        def lcs(X, Y):
            m, n = len(X), len(Y)
            L = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i-1] == Y[j-1]: L[i][j] = L[i-1][j-1] + 1
                    else: L[i][j] = max(L[i-1][j], L[i][j-1])
            return L[m][n]
        ref_tokens = ref_text.lower().split()
        hyp_tokens = hyp_text.lower().split()
        lcs_len = lcs(ref_tokens, hyp_tokens)
        prec = lcs_len / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
        rec = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0
        return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0

    def evaluate_text_pro(self, ref_text, hyp_text):
        emb_score = self.embedding_eval.get_score(ref_text, hyp_text)
        key_score, missed, extra, ref_keys = self.key_eval.get_score_detailed(ref_text, hyp_text)
        bleu_score = self.get_bleu4(ref_text, hyp_text)
        rouge_score = self.get_rouge(ref_text, hyp_text)
        
        ranking_score = 0.15*(rouge_score + bleu_score) + 0.4*key_score + 0.3*emb_score
        
        # 强制保留4位小数
        return {
            'ranking_score': round(ranking_score, 4),
            'emb_score': round(emb_score, 4),
            'key_score': round(key_score, 4),
            'bleu_score': round(bleu_score, 4),
            'rouge_score': round(rouge_score, 4),
            'missed_keywords': missed,
            'extra_keywords': extra,
            'gt_keywords': ref_keys
        }

# --- 3. 辅助函数 ---

def categorize_sample(text):
    """简单分类样本，用于分层分析"""
    text = text.lower()
    if 'no evidence' in text: return 'No Tumor'
    if 'metaplastic' in text: return 'Metaplastic'
    if 'micropapillary' in text: return 'Micropapillary'
    if 'mucinous' in text: return 'Mucinous'
    if 'lobular' in text: return 'Lobular'
    if 'fibroadenoma' in text or 'fibroepithelial' in text: return 'Fibroadenoma'
    if 'papillary' in text or 'papilloma' in text: return 'Papillary'
    if 'dcis' in text or 'ductal carcinoma in situ' in text: return 'DCIS'
    if 'invasive carcinoma' in text: return 'NST'
    return 'Other'

# --- 4. 主程序 ---

def main():
    print("正在初始化专业评分器...")
    evaluator = REG_Evaluator_Pro(EMBEDDING_MODEL, SPACY_MODEL)

    # 加载数据
    try:
        with open(GT_PATH, 'r', encoding='utf-8') as f: gt_data = json.load(f)
        with open(PRED_PATH, 'r', encoding='utf-8') as f: pred_data = json.load(f)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}，请检查路径配置。")
        return

    pred_dict = {item['id']: item['report'] for item in pred_data}
    gt_dict = {item['id']: item['report'] for item in gt_data}
    
    # 筛选交集
    eval_ids = [pid for pid in pred_dict.keys() if pid in gt_dict]
    print(f"实际参与评分样本数: {len(eval_ids)}")
    
    results_list = []
    all_missed_keywords = []
    
    print("开始评分...")
    for wsi_id in tqdm(eval_ids):
        gt_text = gt_dict[wsi_id]
        pred_text = pred_dict[wsi_id]
        
        metrics = evaluator.evaluate_text_pro(gt_text, pred_text)
        
        # 补充元数据
        category = categorize_sample(gt_text)
        
        # 构造易读的字典结构
        entry = {
            'id': wsi_id,
            'category': category,
            '总分_Rank': metrics['ranking_score'],
            '关键词分_Key': metrics['key_score'],
            '语义分_Emb': metrics['emb_score'],
            '短语匹配分_BLEU': metrics['bleu_score'],    # 【新增】 修复 KeyError
            '结构匹配分_ROUGE': metrics['rouge_score'],  # 【新增】 修复 KeyError
            '漏报词 (Missed)': metrics['missed_keywords'],
            '多报词 (Extra)': metrics['extra_keywords'],
            '真值报告': gt_text,
            '预测报告': pred_text
        }
        
        all_missed_keywords.extend(metrics['missed_keywords'])
        results_list.append(entry)
        
    # --- 生成 JSON 详情文件 (Details) ---
    print(f"正在保存详情至 {OUTPUT_DETAILS} ...")
    with open(OUTPUT_DETAILS, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    
    # --- 生成总结报告 (Summary) ---
    df = pd.DataFrame(results_list)
    
    # 准备报告内容
    total_samples = len(df)
    overall_score = df['总分_Rank'].mean()
    
    emb_avg = df['语义分_Emb'].mean()
    key_avg = df['关键词分_Key'].mean()
    bleu_avg = df['短语匹配分_BLEU'].mean()  # 修复后的键名
    rouge_avg = df['结构匹配分_ROUGE'].mean() # 修复后的键名

    report_lines = []
    
    report_lines.append(f"总样本数 (TOTAL SAMPLES): {total_samples}")
    report_lines.append(f"总体评分 (OVERALL RANKING SCORE): {overall_score:.4f}")
    report_lines.append("")
    report_lines.append("指标细分 (METRIC BREAKDOWN):")
    # 使用左对齐确保清爽整洁
    report_lines.append(f"  - Embedding (30%): {emb_avg:.4f} (语义准确度)")
    report_lines.append(f"  - Keywords  (40%): {key_avg:.4f} (术语精确度)")
    report_lines.append(f"  - BLEU-4    (15%): {bleu_avg:.4f} (N-gram 匹配)")
    report_lines.append(f"  - ROUGE-L   (15%): {rouge_avg:.4f} (结构匹配)")
    
    report_lines.append("")
    report_lines.append("-" * 30)
    report_lines.append("各病种表现 (PERFORMANCE BY CATEGORY)")
    report_lines.append("-" * 30)
    
    cat_perf = df.groupby('category')['总分_Rank'].agg(['mean', 'count']).sort_values('mean')
    
    for cat, row in cat_perf.iterrows():
        # 清爽对齐格式: 类别名占20字符 | 数量占10字符 | 分数
        report_lines.append(f"{cat:<20} | 数量: {int(row['count']):<4} | 分数: {row['mean']:.4f}")
    
    report_lines.append("")
    report_lines.append("-" * 30)
    report_lines.append("高频漏报词汇 (TOP 10 MISSED KEYWORDS)")
    report_lines.append("-" * 30)
    
    missed_counts = Counter(all_missed_keywords).most_common(10)
    for word, count in missed_counts:
        report_lines.append(f"漏报 {count:<3} 次: '{word}'")

    report_lines.append("")
    report_lines.append("-" * 30)
    report_lines.append("格式检查 (HEADER CHECK)")
    report_lines.append("-" * 30)
    
    target_header = "Breast, sono-guided core biopsy;"
    correct_count = df['预测报告'].apply(lambda x: x.strip().startswith(target_header)).sum()
    report_lines.append(f"Header 匹配率 ('{target_header}'): {correct_count}/{len(df)} ({correct_count/len(df):.1%})")

    # 写入文件
    with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print(f"\n✅ 评分完成！")
    print(f"1. 详情 (JSON/彩色): {OUTPUT_DETAILS}")
    print(f"2. 总结 (报告):       {OUTPUT_SUMMARY}")
    
    # 控制台同时输出一份清爽的报告
    print("\n" + "="*40)
    print("\n".join(report_lines))
    print("="*40 + "\n")

if __name__ == "__main__":
    main()