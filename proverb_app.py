import os 
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

###################################################
############# PROVERBs EXPERT #####################
###################################################

def get_proverb_and_translation(proverb:str):
    #llm
    groq_llm = ChatGroq(
        groq_api_key = os.environ["GROQ_API_KEY"],
        model_name = "llama-3.1-8b-instant", #Check groq website for other llm ids
        temperature = 0.6  
        ) 
    #Frist chain
    first_template = PromptTemplate(input_variables = ["proverb"], 
                                    template = "Complete this proverb {proverb}, return the complete proverb and nothin more.")
    first_chain = LLMChain(llm = groq_llm, prompt = first_template, output_key = "proverb_completed")

    #Second chain
    second_template = PromptTemplate(input_variables = ["proverb_completed"],
                                     template = """
                                                Translate this proverbe <{proverb_completed}> to french, 
                                                return only one translation and nothing more.
                                                """)
    second_chain = LLMChain(llm = groq_llm, prompt = second_template, output_key = "translation")

    #Sequence 
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
