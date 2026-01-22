from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Ensure Google API key is set
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


# -----------------------------
# State definition
# -----------------------------
class AgentState(TypedDict):
    task: str
    plan: str
    research: str
    output: str


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
    ("system", "You are a planner agent. Break the task into clear steps."),
    ("human", "{input}")
])

research_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research agent. Provide detailed factual reasoning."),
    ("human", "{input}")
])

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a writer agent. Produce a final polished answer."),
    ("human", "{input}")
])


# -----------------------------
# Graph nodes
# -----------------------------
def planner_node(state: AgentState):
    response = llm.invoke(
        planner_prompt.format_messages(
            input=state["task"]
        )
    )
    return {"plan": response.content}


def research_node(state: AgentState):
    response = llm.invoke(
        research_prompt.format_messages(
            input=state["plan"]
        )
    )
    return {"research": response.content}


def writer_node(state: AgentState):
    combined_input = f"""
Task:
{state['task']}

Plan:
{state['plan']}

Research:
{state['research']}
"""
    response = llm.invoke(
        writer_prompt.format_messages(
            input=combined_input
        )
    )
    return {"output": response.content}


# -----------------------------
# Build LangGraph
# -----------------------------
graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("researcher", research_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("planner")

graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", END)

app = graph.compile()


# -----------------------------
# Run graph
# -----------------------------
result = app.invoke({
    "task": "Write a detailed article on the impact of artificial intelligence on modern healthcare."
})

print("\nFINAL OUTPUT:\n")
print(result["output"])
