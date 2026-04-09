import os
import json
from typing import Annotated, List, TypedDict, Literal
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from termcolor import colored
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_embeddings, get_llm, DATA_FOLDER, DB_FOLDER, FILES


# Generic Retry Logic (Provider agnostic)
retry_logic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)


def initialize_vector_dbs():
    embeddings = get_embeddings()
    retrievers = {}
    
    for key in FILES.keys():
        persist_dir = os.path.join(DB_FOLDER, key)

        if os.path.exists(persist_dir):
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            retrievers[key] = vectorstore.as_retriever(search_kwargs={"k": 6})
        else:
            print(colored(f"❌ Error: Database for '{key}' not found!", "red"))
            print(colored(f"⚠️ Please run 'python build_rag.py' first.", "yellow"))
            continue
    
    return retrievers

RETRIEVERS = initialize_vector_dbs()


class AgentState(TypedDict):
    question: str
    documents: str
    generation: str
    search_count: int
    needs_rewrite: str


@retry_logic
def retrieve_node(state: AgentState):
    print(colored("--- 🔍 RETRIEVING ---", "blue"))
    question = state["question"]
    llm = get_llm()

    # Task B: LangGraph - Intelligent Router
    options = ["apple", "tesla", "both", "none"]
    router_prompt = f"""
    Analyze the user question and classify it into one of four categories: {options}.
    
    GUIDELINES:
    - If it mentions 'Apple', use 'apple'.
    - If it mentions 'Tesla', use 'tesla'.
    - If it mentions both or is a comparison, use 'both'.
    - If it mentions neither, use 'none'.
    
    Output ONLY valid JSON: {{"datasource": "..."}}
    User Question: {question}
    """
    try:
        response = llm.invoke(router_prompt)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        res_json = json.loads(content)
        target = res_json.get("datasource", "both")
        if target not in options:
            target = "both"
    except Exception as e:
        print(colored(f"⚠️ Router error: {e}. Defaulting to 'both'.", "yellow"))
        target = "both"

    print(colored(f"🎯 Routing to: {target}", "cyan"))

    docs_content = ""

    if target == "both":
        # Step 2 (NEW): Generate highly specific sub-queries to ensure we get ANNUAL data
        subquery_prompt = f"""
        The user asked: "{question}"
        This question requires data from multiple financial sources: {list(FILES.keys())}.
        
        For each source, write a HIGHLY SPECIFIC retrieval query. 
        Focus on "Full Year 2024", "Annual Report", "12 Months Ended", or "Consolidated Statement of Operations".
        
        Output ONLY valid JSON like:
        {{
            "apple": "Apple 2024 annual total net sales consolidated statement of operations",
            "tesla": "Tesla 2024 full year R&D expenses consolidated income statement"
        }}
        Only include keys for sources in: {list(FILES.keys())}
        """

        try:
            sq_response = llm.invoke(subquery_prompt)
            sq_content = sq_response.content.strip()
            if "```json" in sq_content:
                sq_content = sq_content.split("```json")[1].split("```")[0].strip()
            elif "```" in sq_content:
                sq_content = sq_content.split("```")[1].split("```")[0].strip()
            subqueries = json.loads(sq_content)
            print(colored(f"📝 Sub-queries: {subqueries}", "cyan"))
        except Exception as e:
            print(colored(f"⚠️ Sub-query generation failed: {e}. Falling back to original question.", "yellow"))
            subqueries = {key: question for key in FILES.keys()}

        for key, sub_q in subqueries.items():
            if key in RETRIEVERS:
                docs = RETRIEVERS[key].invoke(sub_q)
                source_name = key.capitalize()
                docs_content += f"\n\n[Source: {source_name}]\n" + "\n".join([d.page_content for d in docs])

    elif target in FILES:
        # Single-source path: unchanged
        if target in RETRIEVERS:
            docs = RETRIEVERS[target].invoke(question)
            source_name = target.capitalize()
            docs_content += f"\n\n[Source: {source_name}]\n" + "\n".join([d.page_content for d in docs])

    return {"documents": docs_content, "search_count": state["search_count"] + 1}

@retry_logic
def grade_documents_node(state: AgentState): 
    print(colored("--- ⚖️ GRADING ---", "yellow"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    # Task C: LangGraph - Relevance Grader (Binary Judge)
    system_prompt = """You are a "Binary Judge." Evaluate the retrieved documents against the user's question.
    
    REQUIREMENT:
    - If the document is relevant to answering the question, output 'yes'.
    - If the document is irrelevant (noise), output 'no'.
    
    CRITICAL: You must answer with ONLY 'yes' or 'no'. No explanation."""

    
    msg = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Retrieved document context: \n\n {documents} \n\n User question: {question}")
    ]
    
    response = llm.invoke(msg)
    content = response.content.strip().lower()
    
    grade = "yes" if "yes" in content else "no"
    print(f"   Relevance Grade: {grade}")
    return {"needs_rewrite": grade}

@retry_logic
def generate_node(state: AgentState):
    print(colored("--- ✍️ GENERATING ---", "green"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm() 
    
    # Task E: LangGraph - Final Generator
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional financial analyst. Synthesize the final answer using the retrieved context.\n\n"
                   "CONSTRAINTS:\n"
                   "1. CITATIONS: Strictly cite sources (e.g., [Source: Apple 10-K]).\n"
                   "2. HONESTY: If the information is missing (even after retries), state 'I don't know' instead of hallucinating.\n"
                   "3. TEMPORAL ACCURACY: Distinguish between '3 Months Ended' and '12 Months Ended'. Use '2024' (Annual) unless instructed otherwise.\n"
                   "4. ENGLISH ONLY: The Final Answer must be in English.\n\n"
                   "Context:\n{context}"),
        ("human", "{question}"),
    ])

    
    chain = prompt | llm
    response = chain.invoke({"context": documents, "question": question})
    return {"generation": response.content}

@retry_logic
def rewrite_node(state: AgentState): 
    print(colored("--- 🔄 REWRITING QUERY ---", "red"))
    question = state["question"]
    llm = get_llm()
    
    # Task D: LangGraph - Query Rewriter
    msg = [ 
        HumanMessage(content=f"The previous search for '{question}' yielded irrelevant results. \n"
                             f"Your goal is to rewrite the original question to be more specific or use better financial terminology. \n"
                             f"Transformation Requirement: Transform vague queries (e.g., 'how much did they spend on new tech') into precise terms (e.g., 'Research and Development expenses').\n"
                             f"Output ONLY the new question text.")
    ]
    response = llm.invoke(msg)
    new_query = response.content.strip()
    print(f"   New Question: {new_query}")
    return {"question": new_query}

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    def decide_to_generate(state):
        if state["needs_rewrite"] == "yes":
            return "generate"
        else:
            if state["search_count"] > 2: 
                print("   (Max retries reached, generating anyway...)")
                return "generate"
            return "rewrite"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        },
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()

def run_graph_agent(question: str):
    app = build_graph()
    inputs = {"question": question, "search_count": 0, "needs_rewrite": "no", "documents": "", "generation": ""}
    # Using stream to see progress if needed, but invoke is fine for simple return
    result = app.invoke(inputs)
    return result["generation"]

# --- Legacy ReAct Agent ---
def run_legacy_agent(question: str):
    print(colored("--- 🤖 RUNNING LEGACY AGENT (ReAct) ---", "magenta"))
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools.retriever import create_retriever_tool
    from langchain.tools.render import render_text_description

    tools = []
    for key, retriever in RETRIEVERS.items():
        tools.append(create_retriever_tool(
            retriever, 
            f"search_{key}_financials", 
            f"Searches {key.capitalize()}'s financial data."
        ))

    if not tools:
        return "System Error: No tools available."

    llm = get_llm()

    # Task A: LangChain ReAct Agent
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

STRUCTURAL REQUIREMENTS:
You must strictly follow the ReAct loop format:
Question: the input question you must answer
Thought: you should always reason about what to do next
Action: the tool to take, should be one of [{tool_names}]
Action Input: the specific query to send to the tool
Observation: the result returned by the tool
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the conclusion

BEHAVIORAL CONSTRAINTS:
1. English Only: The Final Answer must be in English, even if the user asks in Chinese.
2. Year Precision: Distinguish between 2024, 2023, and 2022 columns in financial tables carefully.
3. Honesty: If the exact 2024 figure is not found, state "I don't know" rather than guessing.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5
    )

    try:
        result = agent_executor.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Legacy Agent Error: {e}"