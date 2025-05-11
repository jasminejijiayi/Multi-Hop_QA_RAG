from openai import OpenAI
import json
import re
import string
import collections
import numpy as np
import os
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# --- RAG Components ---
# 配置 OpenAI 客户端
client = OpenAI(base_url="http://47.242.151.133:21123/v1", api_key="abc123")

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

def retrieve_chunks(question, context_chunks, top_k=5):
    """使用 BM25 检索相关块"""
    tokenized_corpus = [chunk.split() for chunk in context_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = question.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [context_chunks[i] for i in top_indices]

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
def run_evaluation(num_samples_to_run=200):
    """加载数据、执行RAG检索、调用模型并评估"""
    dataset = load_dataset(num_samples=num_samples_to_run)
    results = []
    
    for item in tqdm(dataset):
        # 分块与检索
        context_chunks = split_context_into_chunks(item['context'])
        retrieved_chunks = retrieve_chunks(item['question'], context_chunks)
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
        predicted_answer = query_chat_model(messages, max_tokens=32)
        if predicted_answer:
            results.append({
                'id': item['id'],
                'question': item['question'],
                'predicted_answer': predicted_answer,
                'golden_answer': item['answer']
            })
    
    # 评估与保存结果
    calculate_and_print_scores(results)
    save_results_to_json(results, "results/outputs6.json")
    print("Evaluation complete!")

if __name__ == "__main__":
    run_evaluation(num_samples_to_run=200)
