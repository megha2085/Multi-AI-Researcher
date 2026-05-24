import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_groq import ChatGroq
from crewai.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Swapped to Llama 3.3 70B because Groq gives it a massive 12,000 TPM Free Tier Limit!
groq_llm = ChatGroq(
    model="groq/llama-3.3-70b-versatile", 
    max_tokens=1500 
)


@tool("Web Search")
def web_search_tool(query: str) -> str:
    """Search the web for up-to-date information on any topic."""
    search = TavilySearchResults(max_results=1)
    results = search.invoke({"query": query})
    
    # Ultra-strict truncation to guarantee we stay under the 6000 TPM limit
    truncated_results = str(results)[:1200]  
    
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