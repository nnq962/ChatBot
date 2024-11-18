from config import *
from utils import *
from prompt import *
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import  ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent


def load_llm(model_name):
    if model_name == 'models/gemini-1.5-flash-001':
        return ChatGoogleGenerativeAI(model=model_name, api_key=GEMINI_API_KEY)


class GOOGLE_GEMINI_RETRIEVAL():
    def __init__(self,embedding_model,gemini_model,db,google_api_key):
        self.embbeding = GoogleGenerativeAIEmbeddings(model=embedding_model,google_api_key=google_api_key)
        self.model = ChatGoogleGenerativeAI(model=gemini_model, api_key=google_api_key)
        self.retriever = db.as_retriever()
        self.retriever.search_kwargs['distance_metric'] = 'cos'
        self.retriever.search_kwargs['fetch_k'] = 100
        self.retriever.search_kwargs['k'] = 1

    def respond(self,user_input):
        instruction = """
        CONTEXT: /n/n {context}/n
        Question: {question}
        """
        get_prompt(instruction, sys_prompt)
        prompt_template = get_prompt(instruction, sys_prompt)

        llm_prompt = PromptTemplate(template=prompt_template,input_variables=['context','question'])
        chain_type_kwargs = {"prompt": llm_prompt}

        qa = RetrievalQA.from_chain_type(self.model, 
                                    chain_type="stuff",
                                    retriever=self.retriever,
                                    chain_type_kwargs=chain_type_kwargs,
                                    return_source_documents=True)
        return qa({'query': user_input})
    