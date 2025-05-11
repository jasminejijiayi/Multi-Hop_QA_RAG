from openai import OpenAI
import json
import re
import string
import collections
import numpy as np
import os
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

# --- RAG Components ---
# 配置 OpenAI 客户端
client = OpenAI(base_url="http://47.242.151.133:21123/v1", api_key="abc123")

# 加载 sentence-transformers 模型
print("Loading sentence-transformers model...")
SENTENCE_MODEL = "all-MiniLM-L6-v2"  # 小型高效模型
try:
    embedder = SentenceTransformer(SENTENCE_MODEL)
    print(f"Loaded {SENTENCE_MODEL} successfully")
except Exception as e:
    print(f"Error loading sentence-transformers model: {e}")
    print("Falling back to BM25-only retrieval")
    embedder = None

# --- RAG 分块与检索逻辑 ---
def split_context_into_chunks(context, chunk_size=400):
    """将上下文按段落和固定大小分块"""
    passages = re.split(r'Passage \d+:', context)
    passages = [p.strip() for p in passages if p.strip()]
    
    chunks = []
    for passage in passages:
        sentences = re.split(r'(?<=[.!?]) +', passage)
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            sentence_words = sentence.split()
            if current_length + len(sentence_words) > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = sentence_words
                current_length = len(sentence_words)
            else:
                current_chunk.extend(sentence_words)
                current_length += len(sentence_words)
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    return chunks

def get_embeddings(texts: List[str]) -> np.ndarray:
    """使用 sentence-transformers 生成文本嵌入表示"""
    if embedder is None:
        # 如果没有加载成功模型，返回空矩阵
        return np.zeros((len(texts), 1))
    try:
        embeddings = embedder.encode(texts, show_progress_bar=False)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return np.zeros((len(texts), 1))

def retrieve_bm25(question: str, context_chunks: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    """使用 BM25 检索相关块"""
    tokenized_corpus = [chunk.split() for chunk in context_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = question.split()
    scores = bm25.get_scores(tokenized_query)
    
    # 将得分规范化到 [0, 1] 范围
    if max(scores) > 0:
        scores = scores / max(scores)
    
    # 返回得分最高的前 k 个文档对
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(context_chunks[i], scores[i]) for i in top_indices]

def retrieve_vector(question: str, context_chunks: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    """使用向量距离检索相关块"""
    if embedder is None or len(context_chunks) == 0:
        return [(chunk, 0.0) for chunk in context_chunks[:min(top_k, len(context_chunks))]]
    
    # 生成文档和查询的嵌入
    query_embedding = get_embeddings([question])
    chunk_embeddings = get_embeddings(context_chunks)
    
    # 计算余弦相似度
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # 返回得分最高的前 k 个文档对
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(context_chunks[i], similarities[i]) for i in top_indices]

def hybrid_ranking(question: str, context_chunks: List[str], top_k: int = 5, 
                 bm25_weight: float = 0.5, vector_weight: float = 0.5) -> List[str]:
    """结合 BM25 和向量検索的混合检索方法"""
    # 获取 BM25 和向量的检索结果
    bm25_results = retrieve_bm25(question, context_chunks, top_k=top_k*2)
    vector_results = retrieve_vector(question, context_chunks, top_k=top_k*2)
    
    # 合并分数：建立文档到分数的映射
    chunk_scores = {}
    for chunk, score in bm25_results:
        if chunk not in chunk_scores:
            chunk_scores[chunk] = 0
        chunk_scores[chunk] += score * bm25_weight
    
    for chunk, score in vector_results:
        if chunk not in chunk_scores:
            chunk_scores[chunk] = 0
        chunk_scores[chunk] += score * vector_weight
    
    # 按合并后的分数排序
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 返回得分最高的前 k 个文档
    return [chunk for chunk, _ in sorted_chunks[:top_k]]

def retrieve_chunks(question: str, context_chunks: List[str], top_k: int = 5) -> List[str]:
    """使用混合检索方法检索相关块"""
    # 根据 embedder 是否加载成功决定检索方式
    if embedder is None:
        print("Using BM25-only retrieval")
        results = retrieve_bm25(question, context_chunks, top_k)
        return [chunk for chunk, _ in results]
    else:
        # 使用混合检索
        return hybrid_ranking(question, context_chunks, top_k, 
                             bm25_weight=0.7, vector_weight=0.3)  # 可以调整权重

# --- Data Loading ---
def load_dataset(file_path="hotpotqa_longbench.json", num_samples=None):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if num_samples is not None:
        return data[:num_samples]
    return data

# --- Model Interaction ---
def query_chat_model(messages, max_tokens=32):
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",  # 使用 vLLM 服务器上的 Llama3.1-8B-Instruct 模型
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying Llama3.1 model: {e}")
        return None

# --- Evaluation Functions ---
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_em(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()

    if not gold_toks or not pred_toks:
        return int(gold_toks == pred_toks)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# --- Score Calculation and Printing ---
def calculate_and_print_scores(results):
    em_scores = []
    f1_scores = []
    
    print("=" * 80)
    print("Individual Results:")
    print("=" * 80)

    for i, result in enumerate(results):
        gold = result['golden_answer']
        pred = result['predicted_answer']
        
        em = compute_em(gold, pred)
        f1 = compute_f1(gold, pred)
        
        result['em'] = em
        result['f1'] = f1
        em_scores.append(em)
        f1_scores.append(f1)

        print(f"--- Sample {i+1} (ID: {result['id']}) ---")
        print(f"Question: {result['question']}")
        print(f"Predicted Answer: {pred}")
        print(f"Golden Answer: {gold}")
        print(f"EM: {em}")
        print(f"F1: {f1:.4f}")
        print("-" * 40)

    avg_em = np.mean(em_scores) if em_scores else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0

    print("=" * 80)
    print("Overall Scores:")
    print("=" * 80)
    print(f"Average EM: {avg_em:.4f} ({avg_em * 100:.2f}%)")
    print(f"Average F1: {avg_f1:.4f} ({avg_f1 * 100:.2f}%)")
    print("=" * 80)

    return avg_em, avg_f1

# --- Save Results Function ---
def save_results_to_json(results, output_path):
    output_data = [{'id': r['id'], 'pred_answer': r['predicted_answer']} for r in results]

    print(f"Saving results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print("Results saved successfully.")

# --- Main Execution Logic ---
def run_evaluation(num_samples_to_run=200, top_k=5, bm25_weight=0.7, vector_weight=0.3, chunk_size=400):
    """加载数据、执行混合RAG检索、调用模型并评估"""
    print(f"Running evaluation with hybrid retrieval:")
    print(f"- Samples: {num_samples_to_run}")
    print(f"- Top-K: {top_k}")
    print(f"- BM25 weight: {bm25_weight}")
    print(f"- Vector weight: {vector_weight}")
    print(f"- Chunk size: {chunk_size}")
    
    dataset = load_dataset(num_samples=num_samples_to_run)
    results = []
    
    for item in tqdm(dataset):
        # 分块与检索
        context_chunks = split_context_into_chunks(item['context'], chunk_size=chunk_size)
        
        # 混合检索
        if embedder is not None:
            retrieved_chunks = hybrid_ranking(
                item['question'], context_chunks, top_k=top_k,
                bm25_weight=bm25_weight, vector_weight=vector_weight
            )
        else:
            # 如果向量模型加载失败，使用纯 BM25
            bm25_results = retrieve_bm25(item['question'], context_chunks, top_k=top_k)
            retrieved_chunks = [chunk for chunk, _ in bm25_results]
        
        context = "\n".join(retrieved_chunks)
        
        # 构造模型输入
        query_to_model = (
            f"Context: {context}\n"
            f"Question: {item['question']}\n"
            "Instruction: Answer concisely using ONLY the context. "
            "For Yes/No questions, respond 'Yes' or 'No'. "
            "Other answers must be short phrases without full sentences."
        )
        messages = [
            {"role": "system", "content": "You are a factual question-answering assistant."},
            {"role": "user", "content": query_to_model}
        ]
        
        # 调用模型
        predicted_answer = query_chat_model(messages, max_tokens=40)
        if predicted_answer:
            results.append({
                'id': item['id'],
                'question': item['question'],
                'predicted_answer': predicted_answer,
                'golden_answer': item['answer'],
                'retrieval_method': 'hybrid' if embedder is not None else 'bm25'
            })
    
    # 评估与保存结果
    calculate_and_print_scores(results)
    save_results_to_json(results, "results/outputs_hybrid.json")
    print("Evaluation complete!")

if __name__ == "__main__":
    # 可以调整参数进行实验
    run_evaluation(
        num_samples_to_run=200,  # 测试样本数量
        top_k=5,               # 检索结果数量
        bm25_weight=0.8,       # BM25 分数权重
        vector_weight=0.2,     # 向量检索分数权重
        chunk_size=400         # 文本分块大小
    )
