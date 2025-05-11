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

# --- 多跳问题分解策略 ---
def decompose_question(question):
    """将复杂多跳问题分解为子问题"""
    messages = [
        {"role": "system", "content": "You are an AI assistant that decomposes complex questions into simpler sub-questions."},
        {"role": "user", "content": f"""
            I need to decompose this complex question into 2-3 simpler sub-questions that can be answered separately.
            The goal is to break down multi-hop reasoning into individual steps.
            
            Here are some examples:
            
            Complex: "Who directed the film that starred Tom Hanks as a man stranded on an island?"
            Decomposition:
            Q1: "What film starred Tom Hanks as a man stranded on an island?"
            Q2: "Who directed [Answer to Q1]?"
            
            Complex: "What is the population of the city where the headquarters of the company that makes iPhone is located?"
            Decomposition:
            Q1: "What company makes iPhone?"
            Q2: "Where is the headquarters of [Answer to Q1] located?"
            Q3: "What is the population of [Answer to Q2]?"
            
            Now, decompose this question: "{question}"
            
            Format your response strictly as a JSON array of sub-questions, for example:
            ["What is X?", "Where is Y related to X?"]
            Only include the json array in your response, nothing else.
        """}
    ]
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        
        # 解析返回的 JSON 字符串
        response_text = response.choices[0].message.content
        # 处理可能的额外内容，提取有效的 JSON 部分
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
            
        sub_questions = json.loads(response_text)
        return sub_questions
    except Exception as e:
        print(f"Error decomposing question: {e}")
        # 如果分解失败，返回原问题作为单一子问题
        return [question]

def is_multi_hop_question(question):
    """判断是否为多跳问题的启发器"""
    # 基于问题的复杂度判断
    # 包含多个实体的问题可能是多跳的
    entities_indicators = [" of ", " by ", " from ", " in ", " who ", " where ", " when "]
    complex_structure = any(indicator in question.lower() for indicator in entities_indicators)
    
    # 包含关系性词汇的问题可能是多跳的
    relation_indicators = ["related", "associated", "connected", "linked", "same", "common"]
    has_relation_terms = any(indicator in question.lower() for indicator in relation_indicators)
    
    # 包含多个问句的可能是多跳的
    question_words = ["what", "who", "where", "when", "which", "how", "why"]
    question_word_count = sum(1 for word in question_words if question.lower().count(word) >= 1)
    
    # 长度超过一定限度的可能是多跳的
    is_long_question = len(question.split()) > 15
    
    # 综合判断
    return (complex_structure and (has_relation_terms or question_word_count >= 2)) or is_long_question

def combine_results(sub_questions, sub_answers, final_question):
    """合并子问题的答案为最终答案"""
    context = ""
    
    # 构建合并的上下文
    for i, (q, a) in enumerate(zip(sub_questions, sub_answers)):
        if a:
            context += f"Q{i+1}: {q}\nA{i+1}: {a}\n\n"
    
    # 使用模型生成最终答案
    messages = [
        {"role": "system", "content": "You are a factual question-answering assistant."},
        {"role": "user", "content": f"""
            I have broken down a complex question into sub-questions and found answers to each.
            Please use this information to answer the original question concisely.
            
            Original Question: {final_question}
            
            Sub-questions and answers:
            {context}
            
            Based on these answers, provide a very concise final answer to the original question.
            Your answer should be a short phrase or name, not a complete sentence.
            For Yes/No questions, just answer 'Yes' or 'No'.
        """}
    ]
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=messages,
            max_tokens=40,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error combining results: {e}")
        # 如果合并失败，返回最后一个子问题的答案
        return sub_answers[-1] if sub_answers else None

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
        question = item['question']
        context_chunks = split_context_into_chunks(item['context'])
        predicted_answer = None
        
        # 检测是否为多跳问题
        if is_multi_hop_question(question):
            print(f"\nDetected multi-hop question: {question}")
            # 尝试分解问题
            sub_questions = decompose_question(question)
            print(f"Decomposed into: {sub_questions}")
            
            # 处理每个子问题
            sub_answers = []
            for sub_q in sub_questions:
                # 为每个子问题检索相关内容
                sub_retrieved_chunks = retrieve_chunks(sub_q, context_chunks)
                sub_context = "\n".join(sub_retrieved_chunks)
                
                # 定制子问题的提示词
                sub_query_to_model = (
                    f"Context: {sub_context}\n"
                    f"Question: {sub_q}\n"
                    "Instruction: Answer concisely using ONLY the context. "
                    "Your answer should be a short phrase or name, not a complete sentence."
                )
                sub_messages = [
                    {"role": "system", "content": "You are a factual question-answering assistant."},
                    {"role": "user", "content": sub_query_to_model}
                ]
                
                # 获取子问题答案
                sub_answer = query_chat_model(sub_messages, max_tokens=32)
                sub_answers.append(sub_answer)
                print(f"Sub-question: {sub_q}\nSub-answer: {sub_answer}\n")
            
            # 合并子问题答案来生成最终答案
            predicted_answer = combine_results(sub_questions, sub_answers, question)
            print(f"Final answer: {predicted_answer}\n")
        else:
            # 对于非多跳问题，使用标准 RAG 方法
            retrieved_chunks = retrieve_chunks(question, context_chunks)
            context = "\n".join(retrieved_chunks)
            
            # 构造模型输入
            query_to_model = (
                f"Context: {context}\n"
                f"Question: {question}\n"
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
                'question': question,
                'predicted_answer': predicted_answer,
                'golden_answer': item['answer'],
                'is_multi_hop': is_multi_hop_question(question)
            })
    
    # 评估与保存结果
    calculate_and_print_scores(results)
    save_results_to_json(results, "results/outputs_decomposed.json")
    print("Evaluation complete!")

if __name__ == "__main__":
    # 测试单个多跳问题分解
    test_mode = False
    if test_mode:
        test_question = "Which artist is known for his work on Marvel Team-Up and Batman: Son of the Demon?"
        print(f"Testing question decomposition for: {test_question}")
        print(f"Is multi-hop: {is_multi_hop_question(test_question)}")
        sub_questions = decompose_question(test_question)
        print(f"Decomposed into: {sub_questions}")
    else:
        # 运行全量评估
        run_evaluation(num_samples_to_run=200)  # 开始可以运行小样本进行测试
