from dotenv import load_dotenv
from langgraph.checkpoint import memory
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm = ChatGroq(model="openai/gpt-oss-20b",streaming=True)
search = GoogleSerperAPIWrapper()
tools = [search.run]

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(
    model = llm,
    tools = tools,
    checkpointer = st.session_state.memory,
    system_prompt = "You are a helpful assistant that can answer questions using Google Search."
)

### streamlit web interface
st.subheader("Q&A Agent with Google Search")

for msg in st.session_state.history:
    role = msg["role"]
    content = msg["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask a question:")
if(query):
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})
    response = agent.stream(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":"google_search_agent"}},
        stream_mode = "messages"
    )
    
    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""

        for chunk in response:
            message += chunk[0].content
            space.write(message)
        
        st.session_state.history.append({"role":"ai","content":message})
