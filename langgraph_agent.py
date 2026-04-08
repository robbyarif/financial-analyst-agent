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
            retrievers[key] = vectorstore.as_retriever(search_kwargs={"k": 3})
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
    
    # --- [START] Task B: Intelligent Router ---
    options = ["apple", "tesla", "both", "none"]
    router_prompt = f"""
    Analyze the user question and route it to the correct data source.
    You MUST classify the question into one of the following categories: {options}.
    
    CRITICAL: 
    - If the user mentions "Apple", choose "apple".
    - If "Tesla", choose "tesla".
    - If both or comparison, choose "both".
    - If none, choose "none".
    
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
        print(colored(f"⚠️ Error parsing router output: {e}. Defaulting to 'both'.", "yellow"))
        target = "both"
    
    print(colored(f"🎯 Routing to: {target}", "cyan"))
    # --- [END] Task B ---

    docs_content = ""
    targets_to_search = []
    if target == "both":
        targets_to_search = list(FILES.keys())
    elif target in FILES:
        targets_to_search = [target]
    
    for t in targets_to_search:
        if t in RETRIEVERS:
            docs = RETRIEVERS[t].invoke(question)
            source_name = t.capitalize()
            docs_content += f"\n\n[Source: {source_name}]\n" + "\n".join([d.page_content for d in docs])

    return {"documents": docs_content, "search_count": state["search_count"] + 1}

@retry_logic
def grade_documents_node(state: AgentState): 
    print(colored("--- ⚖️ GRADING ---", "yellow"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    # --- [START] Task C: Relevance Grader ---
    system_prompt = """You are a "Binary Judge". Your goal is to evaluate if the retrieved documents are relevant to the user's question.
    - If the document contains information that could help answer the question, output 'yes'.
    - If the document is irrelevant or noise, output 'no'.
    
    CRITICAL: You must answer with ONLY one word: 'yes' or 'no'."""
    # --- [END] Task C ---
    
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
    
    # --- [START] Task E: Final Generator ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional financial analyst. Synthesize the final answer using the retrieved context.
        
        REQUIREMENTS:
        1. English Only: The answer must be in English.
        2. Citations: Strictly cite sources (e.g., [Source: Apple 10-K] or [Source: Tesla]).
        3. Honesty: If the specific figure or information is missing after retrieval, honestly state "I don't know" instead of hallucinating.
        
        Context:
        {context}"""),
        ("human", "{question}"),
    ])
    # --- [END] Task E ---
    
    chain = prompt | llm
    response = chain.invoke({"context": documents, "question": question})
    return {"generation": response.content}

@retry_logic
def rewrite_node(state: AgentState): 
    print(colored("--- 🔄 REWRITING QUERY ---", "red"))
    question = state["question"]
    llm = get_llm()
    
    # --- [START] Task D: Query Rewriter ---
    msg = [ 
        HumanMessage(content=f"""The previous search for '{question}' yielded irrelevant results. 
        Your goal is to rewrite the original question to be more specific or use better financial terminology.
        Transform vague queries (e.g., "how much did they spend on new tech") into precise terms (e.g., "Research and Development expenses").
        
        Output ONLY the new question text.""")
    ]
    # --- [END] Task D ---
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

    # --- [START] Task A: LangChain ReAct Agent ---
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

RELEVANT CONSTRAINTS:
1. English Only: The Final Answer must be in English, even if the user asks in Chinese.
2. Year Precision: Distinguish between 2024, 2023, and 2022 columns in financial tables very carefully.
3. Honesty: If the exact figure (especially for 2024) is not found, say "I don't know" rather than guessing.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question in English

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
    # --- [END] Task A ---

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