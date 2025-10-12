import os 
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv() #Import environement variables 

#Streamlit app
st.set_page_config(page_title = "Chat with advanced groq")

#################################################
############### Advanced chatbot ################
#################################################

# Initialize LLM
groq_llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)

# Define chatbot node
def chatbot_node(state: MessagesState):
    response = groq_llm.invoke(state["messages"])
    return {"messages": [response]}

# Build graph with checkpointer
def build_graph():
    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot", chatbot_node)
    workflow.set_entry_point("chatbot")
    workflow.add_edge("chatbot", END)
    
    # Add memory checkpointer for persistence
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Streamlit app
st.title("Chat with advanced Groq")

# Initialize graph once
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# Initialize thread_id
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"

# Configuration with thread_id
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Get current state from graph
current_state = st.session_state.graph.get_state(config)
messages = current_state.values.get("messages", []) if current_state.values else []

# Display chat history
for msg in messages:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    elif msg.type == "ai":
        st.chat_message("assistant").write(msg.content)

# Chat input
if prompt := st.chat_input("Your message"):
    # Display user message
    st.chat_message("user").write(prompt)
    
    # Invoke graph with new message
    result = st.session_state.graph.invoke(
        {"messages": [("user", prompt)]},
        config
    )
    
    # Display bot response
    st.chat_message("assistant").write(result["messages"][-1].content)
    st.rerun()

#Add start new conversatin button 
with st.sidebar:
    if st.button("Start new conversation"):
        st.session_state.thread_id = uuid.uuid4()
        st.rerun()


