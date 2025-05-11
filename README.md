# 多策略 RAG 问答系统

## 项目介绍

本项目构建了一个先进的基于多策略检索增强生成（Retrieval-Augmented Generation, RAG）的问答系统。该系统整合了三种先进策略：基础 BM25 检索、混合向量检索以及多跳问题分解，并将检索结果输入到 Llama3.1-8B-Instruct 模型生成高质量答案。

## 核心功能

### 1. 基础检索模块 (main.py)

- **高效文本分块**：按段落和句子边界分块，保持语义完整性
- **BM25 词汇检索**：高效关键词匹配，无需预训练向量模型

### 2. 混合检索模块 (mix.py)

- **双引擎检索**：结合 BM25 关键词匹配与 sentence-transformers 语义向量
- **加权融合**：适应性地合并词汇和语义匹配结果，提高召回和精准度

### 3. 多跳问题分解模块 (query.py)

- **问题拆解**：自动将复杂多跳问题分解为简单子问题
- **逐步推理**：分别检索子问题并综合答案，模拟人类思考流程

### 4. 其他功能

- **生成模块**：通过 OpenAI 兼容接口调用 Llama3.1-8B-Instruct 模型
- **标准化评估**：使用精确匹配（EM）和 F1 分数评估系统性能

## 实验结果

我们在 hotpotqa_longbench 数据集上进行了全面测试，每种策略的详细性能如下：

| 策略 | `chunk_size` | `top_k` | `max_tokens` | EM | F1 |
|------------|--------|--------|-------------|-----|-----|
| 基础 BM25 | 300 | 2 | 32 | 31.00% | 44.22% |
| 增大输出长度 | 300 | 2 | 40 | 29.50% | 42.83% |
| 增大分块大小 | 400 | 2 | 32 | 31.00% | 44.69% |
| 增加检索数量 | 400 | 3 | 32 | 32.00% | 46.36% |
| 深度检索 | 400 | 4 | 32 | 34.00% | 47.76% |
| 更广范检索 | 400 | 5 | 32 | **37.00%** | **50.86%** |
| 多跳问题分解 | 400 | 5 | 32 | 32.00% | 42.10% |
| 混合检索方法 | 400 | 5 | 40 | 33.50% | 45.74% |

### 参数配置与性能关系

当前最佳参数配置：

| 参数 | 描述 | 最佳值 | 影响 |
|---------|------|---------|-------|
| `chunk_size` | 文本分块大小 | 400 | 过大会包含不相关信息，过小会碎片化语义 |
| `top_k` | 检索返回的片段数量 | 5 | 过多会引入噪音，过少会丢失信息 |
| `max_tokens` | 模型输出的最大标记数 | 32 | 限制答案长度，保证简洁 |

可根据不同需求调整上述参数。

## 系统架构

1. **分块模块**：`split_context_into_chunks` 函数实现智能分块
2. **检索模块**：`retrieve_chunks` 函数实现 BM25 检索
3. **查询处理模块**：实现了多跳问题分解和处理
4. **生成模块**：通过 OpenAI 兼容接口调用 Llama3.1-8B-Instruct 模型
5. **评估模块**：规范化答案并计算 EM 和 F1 指标

## 未来工作方向

1. **分块策略优化**：探索更智能的语义分块策略
2. **混合检索调优**：根据不同类型的问题自适应调整 BM25 和向量权重
3. **查询分解增强**：改进多跳问题的识别和分解策略，提高分解准确性
4. **集成同步推理**：将混合检索与查询分解策略结合，发挥两者的互补优势
5. **上下文感知提示**：改进提示词工程，将检索上下文的结构化信息加入提示中

## 如何使用

### 环境准备

```bash
pip install -r requirements.txt
```

### 运行系统

```bash
基础RAG： python main.py
多跳问题分解策略： python query.py
混合检索方法： python mix.py
```

默认会处理 hotpotqa_longbench.json 数据集中的 200 个样本，结果将保存在 `results/outputs.json` 文件中。

### 参数调整

可以在 `main.py` 中修改以下参数：

- 分块大小：调整 `split_context_into_chunks` 函数中的 `chunk_size` 参数
- 检索数量：调整 `retrieve_chunks` 函数中的 `top_k` 参数
- 模型输出长度：调整 `query_chat_model` 函数中的 `max_tokens` 参数

## 查询分解模块工作原理

多跳查询分解方法通过以下步骤实现：

1. **多跳问题识别**：使用启发式规则判断查询是否是多跳问题
   - 检测问题中的关系词汇（of、by、from 等）
   - 检测问题中的多个实体提及
   - 评估问题的复杂度和长度

2. **问题分解**：利用 LLM 将复杂问题分解为 2-3 个简单子问题
   - 示例：“演过漫威和蝙蝠侠的艺术家是谁？”
   - 被分解为：Q1: “哪些艺术家演过漫威？”, Q2: “哪些艺术家演过蝙蝠侠？”

3. **分解处理**：针对每个子问题单独检索相关内容并生成中间答案

4. **答案合成**：合并子问题的答案生成最终回答
   - 将各子问题及其答案作为上下文提供给模型
   - 让模型生成最终的简明答案



## 混合检索架构（mix.py）

混合检索方法结合了 BM25 词汇匹配和基于 sentence-transformers 的语义向量相似度检索：

### 实现原理

1. **双检索引擎**：
   - BM25 词汇匹配：擅长处理精确关键词匹配
   - 向量检索：使用 sentence-transformers 捕捉语义相似性

2. **混合排序算法**：

   ```python
   def hybrid_ranking(question, context_chunks, top_k=5, bm25_weight=0.7, vector_weight=0.3):
       # 获取 BM25 和向量的检索结果
       bm25_results = retrieve_bm25(question, context_chunks, top_k=top_k*2)
       vector_results = retrieve_vector(question, context_chunks, top_k=top_k*2)
       
       # 合并分数
       chunk_scores = {}
       for chunk, score in bm25_results:
           if chunk not in chunk_scores:
               chunk_scores[chunk] = 0
           chunk_scores[chunk] += score * bm25_weight
       
       for chunk, score in vector_results:
           if chunk not in chunk_scores:
               chunk_scores[chunk] = 0
           chunk_scores[chunk] += score * vector_weight
   ```

3. **配置灵活性**：
   - BM25 权重（默认 0.8）：控制关键词匹配的重要性
   - 向量权重（默认 0.2）：控制语义相似度的影响
   - 可根据查询类型调整权重（事实型问题强化 BM25，概念型问题强化向量）

## 依赖库

- openai: 用于调用模型 API
- numpy: 用于数值计算
- tqdm: 用于进度显示
- rank-bm25: 实现 BM25 检索算法
- sentence-transformers: 用于生成文本嵌入（混合检索模式）
- scikit-learn: 用于相似度计算（混合检索模式）
