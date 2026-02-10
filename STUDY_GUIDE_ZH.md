# RAG From Scratch 逐代码块精读指南（中文）

> 目标：帮助你“每个代码块都吃透”。
> 范围：`rag_from_scratch_1_to_4.ipynb`、`rag_from_scratch_5_to_9.ipynb`、`rag_from_scratch_10_and_11.ipynb`、`rag_from_scratch_12_to_14.ipynb`、`rag_from_scratch_15_to_18.ipynb`。

---

## 0. 你要先建立的总框架（先背下来）

RAG 主流程统一是 3 步：
1. **Indexing**：加载文档、切分、向量化、存储。
2. **Retrieval**：用户问题进来后，去向量库找相关片段。
3. **Generation**：把“问题 + 检索上下文”喂给 LLM，生成答案。

你看到的所有高级技巧（查询改写、路由、重排、HyDE、Self-RAG、Step-back、Multi-vector、ColBERT）都在这 3 步里的某一步做增强。

---

## 1. Import 库逐个判断：是否过时、适合什么任务

> 结论先说：本仓库整体思路不过时，但**部分 import 路径在新版本 LangChain 生态中已有更推荐写法**。

### 1.1 兼容但建议迁移（重点）

- `from langchain.prompts import ChatPromptTemplate`
  - 现状：很多版本仍可用。
  - 建议：优先 `from langchain_core.prompts import ChatPromptTemplate`。
  - 任务：构建聊天模板、few-shot 模板。

- `from langchain_community.vectorstores import Chroma`
  - 现状：可用。
  - 建议：新项目可考虑独立包 `langchain-chroma`（生态更清晰）。
  - 任务：本地向量数据库检索。

- `from langchain_community.llms import Cohere`
  - 现状：可用。
  - 建议：优先使用 provider 专用包（如 `langchain-cohere`）或 ChatModel 接口。
  - 任务：调用 Cohere 模型进行生成。

- `from langchain_core.pydantic_v1 import BaseModel, Field`
  - 现状：用于兼容旧写法。
  - 建议：新代码优先用 pydantic v2 风格（避免长期锁死在 v1 API 语义）。
  - 任务：结构化输出（Router、Query Analyzer）。

- `retriever.get_relevant_documents(...)`
  - 现状：大量示例仍在用。
  - 建议：逐步迁移到 runnable 风格（如 `retriever.invoke(query)`）。
  - 任务：执行召回。

### 1.2 当前仍常用、适配明确

- `langchain_openai.ChatOpenAI`：聊天模型调用。
- `langchain_openai.OpenAIEmbeddings`：文本向量化。
- `langchain_text_splitters.RecursiveCharacterTextSplitter`：分块切片。
- `langchain_core.output_parsers.StrOutputParser`：把模型输出转字符串。
- `langchain_core.runnables.*`：把步骤串成 pipeline。
- `langchain.load.dumps/loads`：序列化用于多查询去重等技巧。
- `ragatouille.RAGPretrainedModel`：ColBERT 类检索重排实验。

### 1.3 工具与基础库用途速查

- `bs4`：网页 HTML 解析（配合 WebBaseLoader 过滤内容）。
- `tiktoken`：token 统计、上下文预算。
- `numpy`：向量运算与相似度。
- `uuid`：文档块唯一 ID（multi-vector 映射常用）。
- `datetime`：查询里的日期过滤。
- `typing`（`Literal/Optional/Tuple`）：约束结构化字段。

---

## 2. 按 Notebook 拆解：每个代码块做什么 + 你该练什么

> 说明：下面按“代码块（cell）”讲解。每个块都附一个小练习，确保你能真正掌握。

---

## A. `rag_from_scratch_1_to_4.ipynb`（基础 RAG）

### A1. 安装依赖与环境变量
- 作用：安装 LangChain / OpenAI / Chroma 等包，注入 API Key。
- 易错点：Key 写死在 notebook 不安全；建议用 `.env` + `python-dotenv`。
- 小练习：把 `OPENAI_API_KEY` 改成读取 `os.getenv()`，当为空时报错提示。

### A2. 网页加载（WebBaseLoader + bs4 过滤）
- 作用：从网页拉取文本，提取有效正文。
- 逐行关键：
  - 初始化 loader（含 URL/解析参数）。
  - `load()` 返回 `Document[]`，每个有 `page_content` 与 `metadata`。
- 小练习：加一个第二 URL，比较两页文本长度和 metadata 差异。

### A3. Token 统计（tiktoken）
- 作用：估算 chunk 大小是否会超模型上下文。
- 小练习：同一段文本比较 `gpt-3.5` 与 `gpt-4` 编码 token 数。

### A4. Embedding 与余弦相似度（OpenAIEmbeddings + numpy）
- 作用：把文本映射到向量空间，计算语义相近程度。
- 小练习：准备 3 句话（两句语义接近、1句无关），打印相似度矩阵。

### A5. 文本切分（RecursiveCharacterTextSplitter）
- 作用：把长文切成可检索小块，保留 overlap 减少语义断裂。
- 小练习：尝试 `chunk_size=250/500/1000`，观察召回变化。

### A6. 向量库构建（Chroma.from_documents）
- 作用：把 chunks 写入向量库。
- 小练习：输出向量库中文档数量，确认与 split 后数量一致。

### A7. Retriever 召回（get_relevant_documents）
- 作用：输入 query 返回 top-k 文档块。
- 小练习：把 `k` 从 2 改到 6，观察答案质量和冗余度变化。

### A8. 生成链（Prompt + ChatOpenAI + StrOutputParser）
- 作用：把 context 与 question 拼入 prompt 后生成自然语言答案。
- 小练习：让模型在回答末尾追加“证据句原文”。

### A9. LangChain Hub Prompt
- 作用：直接复用社区维护的 RAG Prompt 模板。
- 小练习：把 hub prompt 与自定义 prompt 在同一问题上做 A/B 对比。

---

## B. `rag_from_scratch_5_to_9.ipynb`（查询增强与分解）

### B1. Multi-query（多查询改写）
- 作用：一个用户问题，生成多个子查询，提升召回覆盖率。
- 关键：`dumps/loads` 常用于文档去重（对象可哈希化处理）。
- 小练习：打印每个子查询对应命中的文档标题，做覆盖率统计。

### B2. RAG-Fusion
- 作用：多路检索结果融合排序，减少单 query 偏差。
- 小练习：实现简版“按文档出现次数加权”的融合分数。

### B3. Decomposition（问题分解）
- 作用：复杂问题拆成多个更可检索子问题，再合成最终答案。
- 小练习：把“task decomposition for agents”拆成 3 个子问题并回答。

### B4. 递归/迭代 QA 配对累积
- 作用：后续子问题回答可引用前序 Q/A 结果。
- 小练习：实现 `format_qa_pair`，让每轮输出都可追踪来源。

### B5. Step-back Prompting
- 作用：把具体问题抽象成更一般问题，再回到原问题回答。
- 小练习：为同一问题分别用 normal retrieval 和 step-back retrieval，比较回答完整度。

### B6. HyDE（Hypothetical Document Embeddings）
- 作用：先让 LLM 生成“假想答案文档”，用它做检索查询。
- 小练习：比较直接 query 检索 vs HyDE 检索的 top-3 文档重合度。

---

## C. `rag_from_scratch_10_and_11.ipynb`（路由与结构化检索）

### C1. Router（结构化输出路由）
- 作用：用 `Literal[...]` 约束输出到特定数据源（python/js/golang）。
- 关键：
  - `RouteQuery(BaseModel)` 定义 schema。
  - `llm.with_structured_output(RouteQuery)` 强制结构化。
  - `prompt | structured_llm` 形成 router chain。
- 小练习：新增 `rust_docs` 路由并写 3 个测试问题。

### C2. RunnableLambda 分支执行
- 作用：根据路由结果选择不同下游链。
- 小练习：把 `choose_route` 从字符串返回改成真正子链调用。

### C3. Prompt Router（embedding 路由）
- 作用：比较“问题向量”和“多个 prompt 向量”，选最匹配提示词。
- 小练习：增加第三个 `biology_template`，看能否稳定命中。

### C4. YouTube Loader
- 作用：抓取视频字幕 + 元数据，构成可检索文档。
- 小练习：换一条视频 URL，比较 metadata 字段差异。

### C5. Query Analyzer（带过滤条件）
- 作用：把自然语言问题解析成结构化检索条件（浏览量、日期、时长等）。
- 小练习：给 5 条中文查询，检查哪些字段会被正确填充。

---

## D. `rag_from_scratch_12_to_14.ipynb`（索引策略进阶）

### D1. Parent-Document Retriever
- 作用：检索时用小 chunk，返回时给大 chunk（上下文更完整）。
- 常见组件：`InMemoryByteStore`、`ParentDocumentRetriever`。
- 小练习：对同一 query 比较“普通向量检索”与“父文档检索”的可读性。

### D2. Multi-Vector Retriever
- 作用：同一文档存多种向量表示（标题向量、摘要向量、句子向量）。
- 小练习：为每文档加“人工摘要向量”，观察召回提升是否明显。

### D3. RAGatouille / ColBERT
- 作用：使用后期交互式检索（late interaction）提升语义匹配质量。
- 小练习：对同一问题比较 Chroma top-3 与 ColBERT top-3 的语义贴合度。

---

## E. `rag_from_scratch_15_to_18.ipynb`（重排与压缩）

### E1. Cohere Rerank + ContextualCompressionRetriever
- 作用：先粗召回，再用重排器保留最有用片段，降低噪声。
- 小练习：打印“重排前后文档顺序”，人工评估相关性变化。

### E2. 文档压缩策略
- 作用：在 token 预算内尽量保留关键证据。
- 小练习：设置不同 `top_n`，比较答案正确率与成本。

---

## 3. 真正做到“每个代码块掌握”的训练法（建议你照做）

1. **抄写并口述**：每个 cell 手打一遍，并用一句话说它在 RAG 哪一层（索引/检索/生成）。
2. **最小改动实验**：每个 cell 只改 1 个参数，记录结果差异。
3. **反事实检查**：删掉这个 cell，系统会坏在哪里？
4. **迁移改写**：把旧 import 改成推荐写法，确保结果不变。
5. **自测闭环**：给每个模块写 3 个问题（容易/中等/刁钻），跑对比表。

---

## 4. 你下一步可以做的“硬核掌握计划”（7 天）

- Day1：完成 1_to_4，重点理解 chunk + embedding + 检索。
- Day2：完成 5_to_9，重点理解 query transformation。
- Day3：完成 10_and_11，重点理解路由与结构化输出。
- Day4：完成 12_to_14，重点理解索引设计差异。
- Day5：完成 15_to_18，重点理解 rerank/compress。
- Day6：统一迁移 import 到推荐路径并回归测试。
- Day7：自己实现一个小型“可选路由 + 多查询 + 重排”的 RAG demo。

---

## 4.1 如果你是 Python 几乎零基础：建议拆成 24 节（每节 1 小时）

> 结论：**建议 24 节课**。原因是你不只要学 RAG，还要补齐 Python、环境、调试、API 调用、数据结构和工程习惯。

### 阶段 A（先补 Python，6 节）
- 第 1 节：Python 运行环境、变量、字符串、数字、输入输出。
- 第 2 节：列表、字典、元组、集合，重点是“如何组织文本数据”。
- 第 3 节：流程控制（if/for/while）+ 函数定义。
- 第 4 节：模块与 import、pip、虚拟环境（venv）基础。
- 第 5 节：面向对象最小必备（类、实例、属性），为 `Document`/`Retriever` 做准备。
- 第 6 节：异常排查、日志打印、读取环境变量（`os.getenv`）。

### 阶段 B（RAG 基础，8 节）
- 第 7 节：RAG 总流程（Index/Retrieve/Generate）与 notebook 结构。
- 第 8 节：文档加载（WebBaseLoader）与文本清洗。
- 第 9 节：分块（TextSplitter）与 token 预算（tiktoken）。
- 第 10 节：向量化（Embeddings）+ 相似度直觉。
- 第 11 节：向量库与检索（Chroma + retriever）。
- 第 12 节：基础 RAG 链（Prompt + LLM + OutputParser）。
- 第 13 节：调参实验（chunk_size、k、temperature）。
- 第 14 节：小测：独立完成一个基础 RAG 问答流程。

### 阶段 C（查询增强与路由，6 节）
- 第 15 节：Multi-query 与 RAG-Fusion。
- 第 16 节：问题分解（Decomposition）与答案合成。
- 第 17 节：Step-back 与 HyDE。
- 第 18 节：结构化输出路由（Pydantic + Literal）。
- 第 19 节：Prompt Router（embedding 路由）。
- 第 20 节：小测：对同一问题实现两种检索增强并对比。

### 阶段 D（进阶检索与工程收尾，4 节）
- 第 21 节：Parent Document / Multi-vector 设计思路。
- 第 22 节：Rerank 与压缩检索（CohereRerank / Compression）。
- 第 23 节：评测方法（准确率、覆盖率、延迟、成本）。
- 第 24 节：毕业项目：把 5 个 notebook 的方法组合成 mini RAG 系统。

### 可选压缩版（时间紧）
- 若你每周时间很少，可先走 **16 节压缩版**：
  - Python 补基础 4 节
  - RAG 基础 6 节
  - 查询增强 4 节
  - 总复盘 2 节

---

## 5. 额外提醒（工程实践）

- 不要在 notebook 明文写 API Key。
- 对检索链要固定评测集，不要只看单次主观结果。
- 凡是“提升效果”的技巧，都要同时评估：
  - 正确率 / 覆盖率
  - 延迟
  - token 成本
  - 可解释性（能否给证据）


---

## 附录：全部代码块索引（用于逐块打卡）


### rag_from_scratch_10_and_11.ipynb

- Cell 1: `! pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain youtube-transcript`
- Cell 2: `import os`
- Cell 3: `os.environ['OPENAI_API_KEY'] = <your-api-key>`
- Cell 4: `from typing import Literal`
- Cell 5: `question = """Why doesn't the following code work:`
- Cell 6: `result`
- Cell 7: `result.datasource`
- Cell 8: `def choose_route(result):`
- Cell 9: `full_chain.invoke({"question": question})`
- Cell 10: `from langchain.utils.math import cosine_similarity`
- Cell 11: `from langchain_community.document_loaders import YoutubeLoader`
- Cell 12: `import datetime`
- Cell 13: `from langchain_core.prompts import ChatPromptTemplate`
- Cell 14: `query_analyzer.invoke({"question": "rag from scratch"}).pretty_print()`
- Cell 15: `query_analyzer.invoke(`
- Cell 16: `query_analyzer.invoke(`
- Cell 17: `query_analyzer.invoke(`
- Cell 18: `(empty)`

### rag_from_scratch_12_to_14.ipynb

- Cell 1: `! pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain youtube-transcript`
- Cell 2: `import os`
- Cell 3: `os.environ['OPENAI_API_KEY'] = <your-api-key>`
- Cell 4: `from langchain_community.document_loaders import WebBaseLoader`
- Cell 5: `import uuid`
- Cell 6: `from langchain.storage import InMemoryByteStore`
- Cell 7: `query = "Memory in agents"`
- Cell 8: `retrieved_docs = retriever.get_relevant_documents(query,n_results=1)`
- Cell 9: `! pip install -U ragatouille`
- Cell 10: `from ragatouille import RAGPretrainedModel`
- Cell 11: `import requests`
- Cell 12: `RAG.index(`
- Cell 13: `results = RAG.search(query="What animation studio did Miyazaki found?", k=3)`
- Cell 14: `retriever = RAG.as_langchain_retriever(k=3)`

### rag_from_scratch_15_to_18.ipynb

- Cell 1: `! pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain cohere`
- Cell 2: `import os`
- Cell 3: `os.environ['OPENAI_API_KEY'] = <your-api-key>`
- Cell 4: `#### INDEXING ####`
- Cell 5: `from langchain.prompts import ChatPromptTemplate`
- Cell 6: `from langchain_core.output_parsers import StrOutputParser`
- Cell 7: `from langchain.load import dumps, loads`
- Cell 8: `from operator import itemgetter`
- Cell 9: `from langchain_community.llms import Cohere`
- Cell 10: `from langchain.retrievers.document_compressors import CohereRerank`

### rag_from_scratch_1_to_4.ipynb

- Cell 1: `! pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain`
- Cell 2: `import os`
- Cell 3: `os.environ['OPENAI_API_KEY'] = <your-api-key>`
- Cell 4: `import bs4`
- Cell 5: `# Documents`
- Cell 6: `import tiktoken`
- Cell 7: `from langchain_openai import OpenAIEmbeddings`
- Cell 8: `import numpy as np`
- Cell 9: `#### INDEXING ####`
- Cell 10: `# Split`
- Cell 11: `# Index`
- Cell 12: `# Index`
- Cell 13: `docs = retriever.get_relevant_documents("What is Task Decomposition?")`
- Cell 14: `len(docs)`
- Cell 15: `from langchain_openai import ChatOpenAI`
- Cell 16: `# LLM`
- Cell 17: `# Chain`
- Cell 18: `# Run`
- Cell 19: `from langchain import hub`
- Cell 20: `prompt_hub_rag`
- Cell 21: `from langchain_core.output_parsers import StrOutputParser`
- Cell 22: `(empty)`

### rag_from_scratch_5_to_9.ipynb

- Cell 1: `! pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain`
- Cell 2: `import os`
- Cell 3: `os.environ['OPENAI_API_KEY'] = <your-api-key>`
- Cell 4: `#### INDEXING ####`
- Cell 5: `from langchain.prompts import ChatPromptTemplate`
- Cell 6: `from langchain.load import dumps, loads`
- Cell 7: `from operator import itemgetter`
- Cell 8: `from langchain.prompts import ChatPromptTemplate`
- Cell 9: `from langchain_core.output_parsers import StrOutputParser`
- Cell 10: `from langchain.load import dumps, loads`
- Cell 11: `from langchain_core.runnables import RunnablePassthrough`
- Cell 12: `from langchain.prompts import ChatPromptTemplate`
- Cell 13: `from langchain_openai import ChatOpenAI`
- Cell 14: `questions`
- Cell 15: `# Prompt`
- Cell 16: `from operator import itemgetter`
- Cell 17: `answer`
- Cell 18: `# Answer each sub-question individually `
- Cell 19: `def format_qa_pairs(questions, answers):`
- Cell 20: `# Few Shot Examples`
- Cell 21: `generate_queries_step_back = prompt \| ChatOpenAI(temperature=0) \| StrOutputParser()`
- Cell 22: `# Response prompt `
- Cell 23: `from langchain.prompts import ChatPromptTemplate`
- Cell 24: `# Retrieve`
- Cell 25: `# RAG`
- Cell 26: `(empty)`

## 4.2 只有 Qwen API Key、没有现成文本时：怎么改成可跑通的版本

你这个场景非常典型：
- 只有 `Qwen` 的 API Key；
- 没有本地知识库文本。

### 最推荐的示例形态（先给结论）

**推荐用“公开网页 + Qwen 兼容 OpenAI 接口 + 本地 Chroma”做第一个 RAG**，原因：
1. 不需要你提前准备语料（直接抓公开网页）。
2. 保留了完整 RAG 链路（加载/切分/向量化/检索/生成）。
3. 后续替换成你自己的数据最平滑（只需替换 loader）。

---

### 4.2.1 依赖安装（Qwen 版）

```bash
pip install -U langchain langchain-openai langchain-community langchain-text-splitters chromadb beautifulsoup4
```

---

### 4.2.2 最小可运行代码（Qwen + 网页 RAG）

> 注意：DashScope 提供了 OpenAI 兼容模式，这样可以继续用 `ChatOpenAI` / `OpenAIEmbeddings`。

```python
import os
import bs4

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1) 配置 Qwen（OpenAI 兼容）
os.environ["OPENAI_API_KEY"] = os.getenv("QWEN_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

if not os.environ["OPENAI_API_KEY"]:
    raise ValueError("请先设置环境变量 QWEN_API_KEY")

# 2) 加载公开网页（你不用准备任何本地文本）
urls = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/tutorials/rag/",
]
loader = WebBaseLoader(
    web_paths=tuple(urls),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(class_=("theme-doc-markdown", "theme-doc-content"))
    },
)
docs = loader.load()

# 3) 切分
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = splitter.split_documents(docs)

# 4) 向量化 + 入库（Qwen Embedding 模型名按你的账户可用模型调整）
embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 5) 生成模型（Qwen）
llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0,
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

# 6) RAG Prompt + Chain
prompt = ChatPromptTemplate.from_template(
    """你是一个严谨的问答助手。请仅根据给定上下文回答。

上下文：
{context}

问题：{question}

如果上下文无法支持答案，请明确说“我无法从给定资料中确认”。"""
)

rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 7) 试运行
question = "RAG 的核心步骤是什么？"
answer = rag_chain.invoke({"question": question})
print(answer)
```

---

### 4.2.3 你需要改的只有 3 个地方

1. `QWEN_API_KEY`：你的密钥。
2. `model="qwen-plus"`：可换成你账户可用的 Qwen 模型。
3. `model="text-embedding-v3"`：如果不可用，换成你控制台可用 embedding 模型。

---

### 4.2.4 为什么这是“最好的第一个例子”

- **真实**：不是 toy text，而是真实网页资料。
- **完整**：包含了标准 RAG 全流程。
- **低门槛**：不要求你手动做语料库。
- **可扩展**：之后把 `WebBaseLoader` 换成 PDF / 本地文件 / 数据库即可。

---

### 4.2.5 给零基础的 1 小时课堂安排建议（Qwen 专版）

如果你已经决定使用 Qwen，不妨把前 4 节直接变成“可跑通优先”：
1. 第 1 节：环境配置 + API key + 第一次成功调用 `qwen-plus`。
2. 第 2 节：网页加载与文档切分。
3. 第 3 节：向量库构建与检索。
4. 第 4 节：拼出完整 RAG 链并输出答案。

先跑通，再逐步理解每行代码，比一开始死磕理论更稳。

---
