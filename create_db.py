from config import *
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader



def create_vectordb(embedding_model, csv_file_path, google_api_key=None):
    if google_api_key is not None:
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model,google_api_key=google_api_key)
        loader = CSVLoader(file_path=csv_file_path, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()
        dbsearch = FAISS.from_documents(data,embedding=embeddings)
        dbsearch.save_local(DB_FAISS_PATH)
    else:
        None


create_vectordb(embedding_model=MODEL_EMBEDDING,csv_file_path=CSV_FILE_PATH,google_api_key=GEMINI_API_KEY)

