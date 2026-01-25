from typing import TypedDict, List, Dict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os

# -------------------------------------------------
# Env
# -------------------------------------------------
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# -------------------------------------------------
# Tools & LLM
# -------------------------------------------------
search_tool = DuckDuckGoSearchRun()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# -------------------------------------------------
# State (IMPORTANT FIX HERE)
# -------------------------------------------------
class NewsState(TypedDict):
    topics: str
    topic_list: List[str]
    topic_results: Annotated[Dict[str, List[str]], operator.or_]
    aggregated_news: str
    final_output: str

# -------------------------------------------------
# Nodes
# -------------------------------------------------
def split_topics_node(state: NewsState):
    topics = [t.strip() for t in state["topics"].split(",") if t.strip()]
    return {
        "topic_list": topics,
        "topic_results": {}
    }

def make_topic_fetcher(topic: str):
    def fetch_node(state: NewsState):
        query = f"{topic} news last 24 hours site:news"
        results = search_tool.run(query)

        articles = []
        for line in results.split("\n"):
            if "http" in line:
                articles.append(line.strip())
        print(f"Fetched {len(articles)} articles for topic: {topic}")

        # Each parallel node writes ONLY its own slice
        return {
            "topic_results": {
                topic: articles
            }
        }
    return fetch_node

def aggregate_node(state: NewsState):
    blocks = []

    for topic, articles in state["topic_results"].items():
        blocks.append(f"\n### {topic}\n")
        blocks.extend(articles)

    return {"aggregated_news": "\n".join(blocks)}

fact_check_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional fact-checking journalist. "
     "Remove unverifiable claims and keep only credible information."),
    ("human", "{input}")
])

def fact_checker_node(state: NewsState):
    response = llm.invoke(
        fact_check_prompt.format_messages(
            input=state["aggregated_news"]
        )
    )
    return {"aggregated_news": response.content}

editor_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a news editor.
Produce:
- One clear headline
- 3–5 bullet summary
- Sources with links"""),
    ("human", "{input}")
])

def editor_node(state: NewsState):
    response = llm.invoke(
        editor_prompt.format_messages(
            input=state["aggregated_news"]
        )
    )
    return {"final_output": response.content}

# -------------------------------------------------
# Graph
# -------------------------------------------------
graph = StateGraph(NewsState)

graph.add_node("split_topics", split_topics_node)
graph.add_node("aggregate", aggregate_node)
graph.add_node("fact_check", fact_checker_node)
graph.add_node("editor", editor_node)

graph.add_edge(START, "split_topics")

# -------------------------------------------------
# Parallel fan-out
# -------------------------------------------------
TOPICS = ["Rohit Sharma", "Neymar", "Carlos Alcaraz"]

for topic in TOPICS:
    node_name = f"fetch_{topic.replace(' ', '_').lower()}"
    graph.add_node(node_name, make_topic_fetcher(topic))
    graph.add_edge("split_topics", node_name)
    graph.add_edge(node_name, "aggregate")

# -------------------------------------------------
# Final flow
# -------------------------------------------------
graph.add_edge("aggregate", "fact_check")
graph.add_edge("fact_check", "editor")
graph.add_edge("editor", END)

app = graph.compile()

# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    result = app.invoke({
        "topics": "AI regulation, OpenAI, Google Gemini"
    })

    print("\n📰 PERSONALIZED NEWS DIGEST\n")
    print(result["final_output"])
