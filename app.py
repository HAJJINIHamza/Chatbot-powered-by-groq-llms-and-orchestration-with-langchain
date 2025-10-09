import os 
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv

load_dotenv() #Import environement variables 

############### CHAT WITH GROQ LLM ################
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
#Get user input
input = st.text_input("Input", key ="your question")
#Get llm output
response = invoke_groq_llm(input)
submit = st.button("Get answer")

if submit:
    st.subheader("Groq's answer is :")
    st.write(response)



############# PROVERBs EXPERT #####################
st.header("Use proverb expert to complete you proverbs and get their french translation")
#Get user proverb
input = st.text_input ("Input", key = "proverb")
proverb_submit = st.button("Complete proverb")
