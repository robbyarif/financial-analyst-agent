# Assignment 3 Report: Autonomous Multi-Doc Financial Analyst
**Student Name:** [Your Name]
**NCU ID:** [Your ID]
**Submission Date:** 2026/04/09

---

## 1. Executive Summary
This report presents the implementation and benchmarking of an autonomous multi-document financial analyst using LangChain (ReAct baseline) and LangGraph (State-aware RAG). The system is capable of intelligent routing, binary document grading, query rewriting, and source-cited answer generation.

## 2. Technical Implementation
### Task A: LangChain ReAct Agent
The legacy agent was implemented using `create_react_agent` with a custom prompt enforcing:
- **English-only output.**
- **Year precision** (distinguishing 2024, 2023, 2022).
- **Honesty** ("I don't know" for missing figures).

### Tasks B-E: LangGraph Architectural Workflow
The LangGraph system consists of a cyclic graph with the following nodes:
1. **Intelligent Router**: Classifies questions into ["apple", "tesla", "both", "none"].
2. **Retrieve Node**: Dynamically fetches documents based on the router's decision.
3. **Relevance Grader**: A "Binary Judge" that checks if retrieved docs are relevant.
4. **Query Rewriter**: Transforms vague queries into precise financial terminology.
5. **Final Generator**: Synthesizes the answer with strict citations and English-only constraints.

---

## 3. Benchmarking Results

| Model/Configuration | Agent Mode | Score (PASS/Total) | Notes |
|---------------------|------------|--------------------|-------|
| 1000 Chunks / Multilingual | LangChain | 6/14 | Prone to failure on multi-step reasoning. |
| 1000 Chunks / Multilingual | LangGraph | 8/14 | Self-correction improved accuracy. |
| 1000 Chunks / all-MiniLM-L6-v2 | LangGraph | 9/14 | **Best performance.** Higher retrieval precision. |
| 2000 Chunks / Multilingual | LangGraph | 8/14 | Better context completeness for tables. |

---

## 4. Deep Dive Analysis

### 4.1 Embedding Model Comparison (Task B Requirement)
We compared two models from the `sentence-transformers` family:
1. **paraphrase-multilingual-MiniLM-L12-v2**: Designed for cross-lingual tasks.
2. **all-MiniLM-L6-v2**: Highly optimized for sentence similarity in English.

**Observations:**
- The **`all-MiniLM-L6-v2`** model achieved a higher score (9/14 vs 8/14). 
- **Rationale:** Since the source documents (10-K reports) are in English, the multilingual mapping of the first model added unnecessary noise. `all-MiniLM-L6-v2` captured financial nuances (e.g., "R&D" vs "Research and Development") more accurately.

### 4.2 LangGraph vs LangChain Comparison (Task B Requirement)
**LangChain (ReAct):**
- **Pros:** simple to implement, follows a clear logical loop.
- **Cons:** "Linear thinking" leads to dead ends. If retrieval returns noise, the agent often repeats the same action or hallucinates.

**LangGraph (State-aware):**
- **Pros:** Modular control. The **Relevance Grader** proactively prevents noise from reaching the generator. The **Query Rewriter** enables a second chance with better keywords.
- **Score:** Improved from 6/14 to 8/14 in identical DB environments.
- **User Experience:** The system feels more robust and handles "Traps" (e.g., asking about 2025 targets) much better by simply stating "I don't know" after exhaustive rewriting fails.

### 4.3 Chunk Size Trade-off (Task B Requirement)
We compared `chunk_size=1000` vs `chunk_size=2000`.

**Context Precision (Small Chunks - 1000):**
- Highly precise for specific fact-finding (e.g., "Who signed the report?").
- Reduces background noise (tokens) passed to the LLM.

**Context Completeness (Large Chunks - 2000):**
- Critical for **Balance Sheets** and **Income Tables**.
- **Case Study:** In Test D ("Apple Services Cost"), 1000-character chunks often split the table mid-way, causing the agent to miss the "Services" row if it was on a different chunk. 2000-character chunks kept the table context together, allowing the agent to find the answer without multiple retrievals.

---

## 5. Conclusion
The transition to LangGraph significantly enhanced the agent's reliability and self-correction. While larger chunks improve table understanding, the choice of a high-quality embedding model (like `all-MiniLM-L6-v2`) remains the most impactful factor for query accuracy in financial RAG systems.
