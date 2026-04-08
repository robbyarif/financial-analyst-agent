# 📊 Autonomous Multi-Doc Financial Analyst (LangGraph RAG)

This repository contains an autonomous financial research agent built with **LangGraph** and **LangChain**. It can research Apple and Tesla financial reports, perform cross-document comparisons, and self-correct its search queries to improve accuracy.

## 🚀 Features
- **Intelligent Routing**: Automatically routes questions to Apple, Tesla, or both databases.
- **Binary Grading**: A judge evaluates retrieved document relevance before answering.
- **Query Rewriting**: If initial search results are irrelevant, the agent rephrases the question using professional financial terminology.
- **State-Aware Reasoning**: Uses LangGraph to manage the cyclic flow and "memory" of the search process.
- **ReAct Baseline**: Includes a legacy LangChain ReAct agent for performance benchmarking.

## ⚙️ Setup
1. **Environment**: Python 3.11+
2. **Install Dependencies**: `uv pip install -r requirements.txt` (or use `pip`)
3. **API Keys**: Configure `GOOGLE_API_KEY` in `.env`.

## 📂 Project Structure
- `langgraph_agent.py`: Main agent logic (Nodes, Edges, State).
- `build_rag.py`: ETL pipeline for PDF ingestion and vector indexing.
- `evaluator.py`: Automated benchmarking suite with LLM-as-a-Judge.
- `config.py`: Centralized configuration for models and paths.
- `report.md`: Detailed analysis of benchmarking results (LangGraph vs LangChain, Embedding Models, Chunk Sizes).

## 📊 Benchmarking Results Summary
- **LangGraph vs LangChain**: LangGraph achieved an **8/14** score compared to **6/14** for the legacy agent, primarily due to its ability to recover from failed searches via the Query Rewriter.
- **Embedding Model**: `all-MiniLM-L6-v2` outperformed the multilingual variant by providing better semantic matching for specific financial terms in English.
- **Chunk Size Trade-off**: Large chunks (2000) showed superior context completeness for complex balance sheet tables, while small chunks (1000) provided higher precision for simple fact extraction.

---
**Developed for NCU Agentic AI Course.**
