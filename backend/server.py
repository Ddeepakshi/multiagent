import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "execution.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)
logger = logging.getLogger(__name__)


def write_text_file(filename: str, content: str):
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def planner_agent(state):
    logger.info(f"Planner agent started | Query: {state['user_query']}")
    prompt = f"""
    You are a planning agent in a multi agent AI system.
    Your job is to read the user's query and create a short plan for how the worker agent should answer.
    User query: {state['user_query']}
    Return only a short actionable plan.
    """
    response = llm.invoke(prompt)
    plan = response.content if hasattr(response, "content") else str(response)
    state["plan"] = plan
    state["steps"].append({"agent": "Planner", "output": plan})
    write_text_file("planner_output.txt", plan)
    logger.info(f"Planner output: {plan}")
    return state


def worker_agent(state):
    state["worker_calls"] += 1
    feedback = state.get("review_reason", "")
    logger.info(f"Worker call #{state['worker_calls']}")
    prompt = f"""
    You are a worker agent.
    User query: {state['user_query']}
    Plan: {state['plan']}
    Previous reviewer feedback: {feedback}
    Write the best possible improved response. If reviewer feedback exists, explicitly fix those issues.
    """
    response = llm.invoke(prompt)
    draft = response.content if hasattr(response, "content") else str(response)
    state["draft_response"] = draft
    state["steps"].append({"agent": f"Worker (attempt {state['worker_calls']})", "output": draft})
    write_text_file(f"worker_output_{state['worker_calls']}.txt", draft)
    logger.info(f"Worker output #{state['worker_calls']}: {draft}")
    return state


def reviewer_agent(state):
    state["reviewer_calls"] += 1
    prompt = f"""
    You are a strict reviewer agent.
    User query: {state['user_query']}
    Draft response: {state['draft_response']}
    Check for:
    - concrete examples
    - implementation details
    - tradeoffs
    - clarity
    - actionable recommendations
    If anything is missing, revise.
    Return EXACTLY in this format:
    Decision: approve OR revise
    Reason: brief reason
    """
    response = llm.invoke(prompt)
    raw_output = response.content.strip() if hasattr(response, "content") else str(response)

    decision = "approve" if "decision: approve" in raw_output.lower() else "revise"
    reason_line = next((line for line in raw_output.splitlines() if line.lower().startswith("reason:")), "")
    reason = reason_line.replace("Reason", "").strip() if reason_line else "No reason provided"

    state["review_decision"] = decision
    state["review_reason"] = reason
    state["steps"].append({"agent": f"Reviewer (round {state['reviewer_calls']})", "output": f"{decision}: {reason}"})
    write_text_file(f"reviewer_output_{state['reviewer_calls']}.txt", raw_output)
    logger.info(f"Reviewer #{state['reviewer_calls']}: {decision} ({reason})")
    return state


def review_router(state):
    if state.get("review_decision") == "approve" or state.get("reviewer_calls", 0) >= 2:
        return "__end__"
    state["revision_count"] = state.get("revision_count", 0) + 1
    return "worker_agent"


workflow = StateGraph(dict)
workflow.add_node("planner_agent", planner_agent)
workflow.add_node("worker_agent", worker_agent)
workflow.add_node("reviewer_agent", reviewer_agent)
workflow.set_entry_point("planner_agent")
workflow.add_edge("planner_agent", "worker_agent")
workflow.add_edge("worker_agent", "reviewer_agent")
workflow.add_conditional_edges(
    "reviewer_agent", review_router,
    {"worker_agent": "worker_agent", "__end__": END}
)
graph = workflow.compile()

# --- FastAPI app ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.post("/run")
def run_query(req: QueryRequest):
    initial_state = {
        "user_query": req.query,
        "plan": "",
        "draft_response": "",
        "review_reason": "",
        "review_decision": "",
        "worker_calls": 0,
        "reviewer_calls": 0,
        "revision_count": 0,
        "steps": [],
    }
    result = graph.invoke(initial_state)
    final_output = result.get("draft_response", "")
    write_text_file("final_output.txt", final_output)
    logger.info(f"Final output: {final_output}")
    return {
        "response": final_output,
        "steps": result.get("steps", []),
    }
