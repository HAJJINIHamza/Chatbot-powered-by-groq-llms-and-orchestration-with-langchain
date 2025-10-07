import os 
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv

load_dotenv() #Import environement variables 


def invoke_groq_llm(prompt:str):
    groq_llm = ChatGroq(
        groq_api_key = os.environ["GROQ_API_KEY"],
        model_name = "llama-3.1-8b-instant", #Check groq website for other llm ids
        temperature = 0.6  
        ) 
    reponse = groq_llm.predict(prompt)
    return reponse

#Streamlit app
st.set_page_config(page_title = "Get answers to you question from groq")
st.header("Chat with groq")
input = st.text_input("Input", key ="your question")
response = invoke_groq_llm(input)
submit = st.button("Get answer")

if submit:
    st.subheader("Groq's answer is :")
    st.write(response)

