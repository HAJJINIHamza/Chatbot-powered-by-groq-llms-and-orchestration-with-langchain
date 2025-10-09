import os 
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
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
input = st.text_input("Your question", key ="input")
#Get llm output
response = invoke_groq_llm(input)
submit = st.button("Get answer")

if submit:
    st.subheader("Groq's answer is :")
    st.write(response)

#Separator
st.markdown("---")

############# PROVERBs EXPERT #####################

def get_proverb_and_translation(proverb:str):
    #llm
    print ("1")
    groq_llm = ChatGroq(
        groq_api_key = os.environ["GROQ_API_KEY"],
        model_name = "llama-3.1-8b-instant", #Check groq website for other llm ids
        temperature = 0.6  
        ) 
    #Frist chain
    print ("2")
    first_template = PromptTemplate(input_variables = ["proverb"], 
                                    template = "Complete this proverb {proverb}, return the complete proverb and nothin more.")
    first_chain = LLMChain(llm = groq_llm, prompt = first_template, output_key = "proverb_completed")

    #Second chain
    print ("3")
    second_template = PromptTemplate(input_variables = ["proverb_completed"],
                                     template = """
                                                Translate this proverbe <{proverb_completed}> to french, 
                                                return only one translation and nothing more.
                                                """)
    second_chain = LLMChain(llm = groq_llm, prompt = second_template, output_key = "translation")

    #Sequence 
    print ("4")
    sequence_chain = SequentialChain(chains = [first_chain, second_chain],
                                     input_variables = ["proverb"],
                                     output_variables = ["proverb_completed", "translation"])
    output = sequence_chain({"proverb": proverb})
    return output
    

st.header("Use proverb expert to complete you proverbs and get their french translation")
#Get user proverb
proverb = st.text_input ("Proverb", key = "proverb")
#Get proverb completed and translation
proverb_completed = get_proverb_and_translation(proverb)
#
proverb_submit = st.button("Complete proverb")
#
if proverb_submit:
    st.write(proverb_completed)
