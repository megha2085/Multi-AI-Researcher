import os
from dotenv import load_dotenv
import streamlit as st
from src.multi_agent_researcher.crew import MultiAgentResearcherCrew

# Load API keys
load_dotenv() 

st.set_page_config(page_title="AI Researcher", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# 1. THE "PRO" CSS INJECTION
st.markdown("""
    <style>
        /* Import modern font (Inter) */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }
        
        /* Clean up massive default top padding */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Hide Streamlit branding */
        #MainMenu, header, footer {visibility: hidden;}
        
        /* Modern Sidebar */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC;
            border-right: 1px solid #E2E8F0;
        }
        
        /* Premium Button */
        .stButton>button {
            background: #0F172A; /* Slate 900 */
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: 1px solid #0F172A;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: all 0.2s ease;
            width: 100%;
        }
        .stButton>button:hover {
            background: #1E293B; /* Slate 800 */
            border: 1px solid #1E293B;
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            color: white;
        }
        
        /* Sleek Input Box */
        .stTextArea textarea {
            border-radius: 8px;
            border: 1px solid #CBD5E1;
            transition: border-color 0.2s ease;
            font-size: 14px;
        }
        .stTextArea textarea:focus {
            border-color: #0F172A;
            box-shadow: 0 0 0 1px #0F172A;
        }
    </style>
""", unsafe_allow_html=True)

# 2. SIDEBAR (Minimalist)
with st.sidebar:
    # Using raw HTML for tighter spacing control
    st.markdown("<h2 style='font-weight: 800; color: #0F172A; margin-bottom: 0; letter-spacing: -0.5px;'>Agentic Research</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B; font-size: 0.85rem; margin-top: 2px;'>Powered by CrewAI & Meta Llama-3</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    topic = st.text_area(
        "Research Directive", 
        value="The impact of Agentic AI on cloud computing",
        height=120,
        help="Specify exactly what you want the agents to uncover."
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    start_button = st.button("Initialize Agents →")

# 3. MAIN AREA (Clean typography)
st.markdown("<h1 style='font-weight: 800; color: #0F172A; letter-spacing: -1.5px; margin-bottom: 0.5rem;'>Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #475569; max-width: 800px; line-height: 1.6;'>Deploy an autonomous team of specialized AI agents to conduct deep web research and synthesize technical reports in real-time.</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: none; height: 1px; background-color: #E2E8F0; margin: 2rem 0;'>", unsafe_allow_html=True)

result_container = st.container()

# 4. EXECUTION LOGIC
if start_button:
    with result_container:
        with st.status("Deploying Agent Swarm...", expanded=True) as status:
            st.write("↳ 📡 Booting Groq inference engine")
            st.write("↳ 🔍 Researcher traversing web sources")
            st.write("↳ ✍️ Writer drafting synthesis")
            
            inputs = {'topic': topic}
            try:
                my_crew = MultiAgentResearcherCrew().crew()
                result = my_crew.kickoff(inputs=inputs)
                status.update(label="Report Finalized", state="complete", expanded=False)
                
                st.markdown("<h3 style='font-weight: 700; color: #0F172A; margin-top: 1rem;'>Final Synthesis</h3>", unsafe_allow_html=True)
                st.markdown(result)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="↓ Download Markdown",
                    data=str(result),
                    file_name=f"{topic.replace(' ', '_')}_report.md",
                    mime="text/markdown"
                )
            except Exception as e:
                status.update(label="System Error", state="error", expanded=True)
                st.error(f"Process terminated: {e}")