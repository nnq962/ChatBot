from prompt import *
from config import *
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
from datasets import load_dataset
from pandasai.responses.response_parser import ResponseParser
from langchain_community.vectorstores import FAISS


# def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
#     SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
#     prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
#     return prompt_template

def get_prompt(instruction, new_system_prompt):
    prompt_template = new_system_prompt + instruction 
    return prompt_template

def get_dataset(name_dataset):
    dataset = load_dataset(name_dataset)
    return dataset

def covert_csv_file(dataset):
    return dataset['train']

def load_faiss_index(db_faiss_path,embeddings):
    db = FAISS.load_local(db_faiss_path,embeddings,allow_dangerous_deserialization=True)
    return db



