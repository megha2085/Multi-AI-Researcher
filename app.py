import os
from dotenv import load_dotenv

# Load API keys from the .env file
load_dotenv() 

import streamlit as st
from src.multi_agent_researcher.crew import MultiAgentResearcherCrew

# Optional: Set the page layout
st.set_page_config(page_title="AI Researcher", page_icon="🤖")

st.title("🤖 Multi-Agent AI Researcher")
st.write("Enter a topic, and my autonomous AI team will build a comprehensive report using Llama-3 and live web search.")

topic = st.text_input("Topic for Research:", "The impact of Agentic AI on cloud computing")

# --- YOUR NEW UPGRADED UI CODE STARTS HERE ---

if st.button("Start AI Agents"):
    # Keep the messy terminal thinking hidden inside a clean drop-down!
    with st.status("🤖 Agents are researching and writing...", expanded=True) as status:
        st.write("Researcher is searching the web...")
        inputs = {'topic': topic}
        my_crew = MultiAgentResearcherCrew().crew()
        result = my_crew.kickoff(inputs=inputs)
        status.update(label="Report Complete!", state="complete", expanded=False)
        
    st.success("Here is your final report:")
    st.markdown(result)
    
    # The ultimate professional touch: A download button
    st.download_button(
        label="📄 Download Report as Markdown",
        data=str(result),
        file_name=f"{topic.replace(' ', '_')}_report.md",
        mime="text/markdown"
    )