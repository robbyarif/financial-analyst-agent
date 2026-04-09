# 📊 Autonomous Multi-Doc Financial Analyst

**Student Name:** Robby Arifandri  
**Student ID:** 114522602  
**Course:** CE8014 Agentic AI | National Central University

This project implements a state-aware RAG (Retrieval-Augmented Generation) system designed to act as an autonomous financial analyst. It is capable of intelligently routing queries between multiple corporate reports (Apple & Tesla), grading document relevance to prevent hallucinations, and iteratively rewriting queries to ensure high-precision data extraction from complex financial tables.

---

## 🏛️ System Architecture

The system supports two distinct agentic architectures for comparative benchmarking:

### 1. LangGraph State-aware Agent (Advanced)
A cyclic directed graph implementation that includes self-correction loops:
*   **Task B: Intelligent Router**: Classifies questions into `["apple", "tesla", "both", "none"]` and generates targeted sub-queries.
*   **Task C: Relevance Grader**: A "Binary Judge" that evaluates if retrieved snippets contain the specific metrics and time periods requested.
*   **Task D: Query Rewriter**: Automatically transforms vague queries (e.g., "new tech spend") into precise financial terms (e.g., "Research and Development expenses") if initial retrieval fails.
*   **Task E: Final Generator**: Synthesizes structured English answers with strict source citations and temporal awareness (detecting 3-month vs. 12-month periods).

### 2. LangChain ReAct Agent (Baseline)
*   **Task A: Legacy ReAct**: A linear reasoning chain using a custom prompt toEstablish a baseline performance for the ReAct loop.

---

## 📊 Benchmarking Highlights

| Configuration | Embedding Model | Score | Success Rate |
| :--- | :--- | :--- | :--- |
| **LangGraph (Peak)** | `paraphrase-multilingual-L12` | **11/14** | **79%** |
| **LangChain (Peak)** | `all-MiniLM-L6-v2` | **11/14** | **79%** |

*Note: Both peak configurations were limited by ground truth discrepancies in the evaluator targets for Tesla financial figures.*

### 🔍 Key Learning: Context Completeness
We verified that a **`chunk_size` of 2000** is optimal for preserving the integrity of large financial tables (like Balance Sheets). This ensures **Context Completeness**, allowing the agent to simultaneously view row labels and all relevant yearly columns (2024, 2023, 2022).

---

## ⚙️ Environment Setup

### 1. Prerequisites
*   **Python 3.11** (Required for compatibility)
*   **Google Gemini API Key** (Set in `.env`)

### 2. Installation
```powershell
# Create & Activate Virtual Environment
python -m venv venv
.\venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 3. Usage
1.  **Build Database**: `python build_rag.py` (Converts PDFs to vector embeddings).
2.  **Run Evaluation**: `python evaluator.py` (Benchmarks the agent).
    *   *Tip: Change `TEST_MODE` in evaluator.py to toggle between "GRAPH" and "LEGACY".*

---

## 📁 Project Structure
*   `langgraph_agent.py`: Core logic for all agent nodes and tools.
*   `build_rag.py`: ETL pipeline for PDF ingestion and vector storage.
*   `evaluator.py`: Suite of 14 complex financial queries for performance scoring.
*   `config.py`: model and path configurations.
*   `report.pdf`: Detailed technical analysis and diagnostic findings.
