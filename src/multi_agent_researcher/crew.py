import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_groq import ChatGroq
from crewai.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Swapped to Gemma-2 for the 15,000 TPM free-tier limit!
groq_llm = ChatGroq(
    model="groq/gemma2-9b-it", 
    max_tokens=1000 
)

@tool("Web Search")
def web_search_tool(query: str) -> str:
    """Search the web for up-to-date information on any topic."""
    search = TavilySearchResults(max_results=1)
    results = search.invoke({"query": query})
    
    # Aggressive truncation (down to roughly 250 tokens)
    truncated_results = str(results)[:1000]  
    
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