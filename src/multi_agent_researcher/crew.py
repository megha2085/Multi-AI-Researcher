import time
import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# ---------------------------------------------------------
# THE HACKER FIX: Monkey-patching CrewAI's active bug
# This intercepts the bad function and neutralizes it.
# ---------------------------------------------------------
# ---------------------------------------------------------

# Initialize the native CrewAI LLM wrapper
groq_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)

@tool("Web Search")
def web_search_tool(query: str) -> str:
    """Search the web for up-to-date information on any topic."""
    search = TavilySearchResults(max_results=1)
    results = search.invoke({"query": query})
    
    # 1. Aggressive truncation (down to roughly 250 tokens)
    truncated_results = str(results)[:1000] 
    
    # 2. The Free-Tier Hack: Force the pipeline to pause to let Groq's limit reset!
    time.sleep(20) 
    
    return truncated_results

@CrewBase
class MultiAgentResearcherCrew():
    """MultiAgentResearcher crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[web_search_tool],
            llm=groq_llm, 
            verbose=True
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer'],
            llm=groq_llm, 
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config['research_task'])

    @task
    def writing_task(self) -> Task:
        return Task(config=self.tasks_config['writing_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )