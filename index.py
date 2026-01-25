from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import re

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


# -----------------------------
# State
# -----------------------------
class AgentState(TypedDict):
    task: str
    plan: List[str]
    tech_research: str
    example_research: str
    draft: str
    critique: str
    score: int
    final_output: str


# -----------------------------
# LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# -----------------------------
# Prompts
# -----------------------------
planner_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a planner agent. Break the task into numbered steps as a list."),
    ("human", "{input}")
])

tech_research_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a technical research agent. Explain internal mechanisms, APIs, and architecture."),
    ("human", "{input}")
])

example_research_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an example-focused research agent. Provide concrete, practical examples."),
    ("human", "{input}")
])

writer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a writer. Combine all research into a clear, structured explanation."),
    ("human", "{input}")
])

critic_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict critic. Score the answer from 1–10 and explain weaknesses."),
    ("human", "{input}")
])


# -----------------------------
# Nodes
# -----------------------------
def planner_node(state: AgentState):
    response = llm.invoke(
        planner_prompt.format_messages(input=state["task"])
    )
    steps = re.findall(r"\d+\.\s*(.*)", response.content)
    return {"plan": steps}


def tech_research_node(state: AgentState):
    response = llm.invoke(
        tech_research_prompt.format_messages(
            input="\n".join(state["plan"])
        )
    )
    return {"tech_research": response.content}


def example_research_node(state: AgentState):
    response = llm.invoke(
        example_research_prompt.format_messages(
            input="\n".join(state["plan"])
        )
    )
    return {"example_research": response.content}


def writer_node(state: AgentState):
    combined = f"""
Task:
{state['task']}

Technical Research:
{state['tech_research']}

Examples:
{state['example_research']}
"""
    response = llm.invoke(
        writer_prompt.format_messages(input=combined)
    )
    return {"draft": response.content}


def critic_node(state: AgentState):
    response = llm.invoke(
        critic_prompt.format_messages(input=state["draft"])
    )
    score_match = re.search(r"(\d{1,2})/10", response.content)
    score = int(score_match.group(1)) if score_match else 5
    return {
        "critique": response.content,
        "score": score
    }


def rewrite_node(state: AgentState):
    improved_input = f"""
Original Draft:
{state['draft']}

Critique:
{state['critique']}

Rewrite the answer addressing all issues.
"""
    response = llm.invoke(improved_input)
    return {"final_output": response.content}


def accept_node(state: AgentState):
    return {"final_output": state["draft"]}


# -----------------------------
# Conditional routing
# -----------------------------
def should_rewrite(state: AgentState):
    return "rewrite" if state["score"] < 8 else "accept"


# -----------------------------
# Build graph
# -----------------------------
graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("tech_research", tech_research_node)
graph.add_node("example_research", example_research_node)
graph.add_node("writer", writer_node)
graph.add_node("critic", critic_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("accept", accept_node)

graph.set_entry_point("planner")

# Parallel research
graph.add_edge("planner", "tech_research")
graph.add_edge("planner", "example_research")

# Join into writer
graph.add_edge("tech_research", "writer")
graph.add_edge("example_research", "writer")

graph.add_edge("writer", "critic")

graph.add_conditional_edges(
    "critic",
    should_rewrite,
    {
        "rewrite": "rewrite",
        "accept": "accept"
    }
)

graph.add_edge("rewrite", END)
graph.add_edge("accept", END)

app = graph.compile()


# -----------------------------
# Run
# -----------------------------
result = app.invoke({
    "task": "Explain LangGraph and its use cases with an example"
})

print("\nFINAL OUTPUT:\n")
print(result["final_output"])
