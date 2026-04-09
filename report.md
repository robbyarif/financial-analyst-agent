---
title: "Assignment 3 Report: Autonomous Multi-Doc Financial Analyst"
---

<title>Assignment 3 Report: Autonomous Multi-Doc Financial Analyst</title>

# Assignment 3 Report: Autonomous Multi-Doc Financial Analyst
**Student Name:** Robby Arifandri  
**Student ID:** 114522602

This assignment is part of the course CE8014 Agentic AI at National Central University, Spring 2026.

---

## 1. Summary
This report presents the implementation and benchmarking of an autonomous multi-document financial analyst using LangChain (ReAct baseline) and LangGraph (State-aware RAG). After multiple optimization rounds involving chunk size adjustments, prompt engineering, and embedding model comparisons, we achieved a peak performance of **11/14 (79%)**.

## 2. Technical Implementation
### Task A: LangChain ReAct Agent
The legacy agent was implemented using `create_react_agent` with a custom prompt enforcing English-only output and year precision.

### Tasks B-E: LangGraph Architectural Workflow
The LangGraph system consists of a cyclic graph with the following nodes:
1. **Intelligent Router**: Classifies questions into companies.
2. **Retrieve Node**: Dynamically fetches documents using company-specific sub-queries.
3. **Relevance Grader**: Binary Judge for document relevance.
4. **Query Rewriter**: Iterative keyword refinement.
5. **Final Generator**: Synthesizes answers with temporal precision (Annual vs. Quarterly) and source citations.

---

## 3. Benchmarking Results (Comparative Analysis)

| Embedding Model | Agent Mode | Score (PASS/Total) | Notes |
|-----------------|------------|--------------------|-------|
| Multilingual-L12 | LangChain | 6/14 | Initial baseline (Assignment 3 starter). |
| Multilingual-L12 | LangGraph | 11/14 | Best Graph performance with high complexity. |
| all-MiniLM-L6-v2 | LangGraph | 9/14 | Lower accuracy with reduced model depth. |
| **all-MiniLM-L6-v2** | **LangChain** | **11/14** | **Legacy agent performance matched peak graph score.** |

---

## 4. Deep Dive Analysis

### 4.1 Embedding Model Comparison: Multilingual-L12 vs. English-L6
We explicitly compared two models:
1. **paraphrase-multilingual-MiniLM-L12-v2**: 12 layers, 384 dimensions.
2. **all-MiniLM-L6-v2**: 6 layers, 384 dimensions.

**Findings:**
- **Graph Complexity vs. Model Depth**: In the LangGraph setup, the **Multilingual-L12** model (11/14) outperformed the English **L6-v2** (9/14). The deeper model better supports the complex reasoning required for our state-aware re-ranking and rewriting logic.
- **English Optimization in Legacy Agent**: Surprisingly, the Legacy ReAct agent achieved **11/14** when using the English-optimized **all-MiniLM-L6-v2**. This suggests that for simpler linear workflows, high-quality, task-specific (English) embeddings can compensate for a simpler agent architecture.
- **Consistency**: Both peak performers (Graph+L12 and Legacy+L6) achieved the exact same final score, limited by the same 3 Tesla ground truth discrepancies.

### 4.2 Chunk Size Trade-off: Precision vs. Completeness
We experimented with different `chunk_size` values to find the optimal balance for financial document analysis, specifically targeting large, multi-column tables like the **Balance Sheet**.

#### Context Precision (Small Chunks: e.g., 500-1000 characters)
- **Definition**: Minimizes irrelevant noise by ensuring the retrieved context is focused strictly on the query keywords.
- **Impact**: While precision is high for simple fact extraction, small chunks frequently "cut" financial tables mid-row or mid-column. This leads to failures in questions requiring comparative data (e.g., comparing 2024 vs 2023 figures) because the relevant columns for the target year are often stranded in an adjacent, un-retrieved chunk.

#### Context Completeness (Large Chunks: e.g., 2000+ characters)
- **Definition**: Ensures the entire structural unit (like a full table or a long disclosure note) is preserved in a single retrieval block.
- **Impact**: By using **`chunk_size=2000`** with a **400-character overlap**, we prioritize **Context Completeness**. This is critical for answering questions about the Balance Sheet or Statement of Operations (like Apple's "Services" cost of sales), as it guarantees that both the row labels and all relevant year columns (2024, 2023, 2022) are visible to the LLM simultaneously.

**The Trade-off**: We chose `chunk_size=2000` because in financial RAG, completeness of structural data (tables) is more valuable than the minor "noise" introduced from surrounding text. Reducing the size would significantly degrade the agent's ability to locate specific metrics within large tables.

### 4.3 Diagnostic Findings: Ground Truth Discrepancies
Our diagnostics confirmed that the remaining 3 "failures" in the 11/14 run (for Tesla R&D, Energy Revenue, and Automotive Sales) are **correct agent answers** based on the provided `tsla-20241231-gen.pdf`. The evaluator appears to have ground truth targets from a different document version or year.

- **Example**: Agent found **$4,540 M** for Tesla R&D (Correct per PDF), but evaluator expected **$4,772 M**.

---

## 5. Conclusion
The LangGraph architecture with **Multilingual-L12 embeddings** and **2000-character chunks** provided the most reliable RAG performance. This configuration consistently avoids temporal ambiguity (Quarterly vs. Annual) and maintains relationship integrity within large financial tables.
