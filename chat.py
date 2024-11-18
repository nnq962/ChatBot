import os 
from llm import *
import pandas as pd
from utils import *
import streamlit as st
from pandasai import Agent
import pandasai
from langchain_core.messages import AIMessage, HumanMessage 
from pandasai.responses.streamlit_response import StreamlitResponse

os.environ["PANDASAI_API_KEY"] = "$2a$10$HAqhJYZyXzqMIHk8bo8AzeioDcIiRte9MzYJ3Mlw5GuUjj5cV6EXO"
embedding = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDING,google_api_key=GEMINI_API_KEY)
db = load_faiss_index(DB_FAISS_PATH,embedding)
qa_rag = GOOGLE_GEMINI_RETRIEVAL(embedding_model=MODEL_EMBEDDING,gemini_model=GEMINI_MODEL,db=db,google_api_key=GEMINI_API_KEY)
llm = load_llm('models/gemini-1.5-flash-001')


st.set_page_config(page_title="ChatBot 3HINC",page_icon="🤖")
st.title("3HINC ChatBot")
with st.sidebar:
    uploaded_file = st.file_uploader("Tải tệp csv của bạn lên đây",type='csv')


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content='xin chào tôi có thể giúp gì bạn')
    ]

#user input 
user_query = st.chat_input('Nhập gì đó')
if user_query is not None and user_query != "" and uploaded_file is None:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    response = str(qa_rag.respond(user_query)['result'])
    st.session_state.chat_history.append(AIMessage(content=response))

if user_query is not None and user_query != "" and uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write("Dữ liệu bạn đã tải lên: ", dataframe.head())
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    agent = Agent(dataframe,config={"verbose": True, "response_parser": StreamlitResponse})
    with st.spinner():
        response = agent.chat(user_query)
        if os.path.isfile('exports/charts/temp_chart.png'):
            im = plt.imread('exports/charts/temp_chart.png')
            st.image(im)
            os.remove('exports/charts/temp_chart.png')
        st.session_state.chat_history.append(AIMessage(content=response))

# conversation
for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message('Human'):
             st.write(message.content)

