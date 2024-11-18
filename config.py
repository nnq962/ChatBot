from dotenv import load_dotenv
import os 
load_dotenv()
# API KEY
ELEVEN_API_KEY = os.getenv('ELEVEN_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
PANDASAI_API_KEY = os.getenv('PANDASAI_API_KEY')

# GOOGLE SERVICES
MODEL_EMBEDDING = 'models/text-embedding-004'
GEMINI_MODEL = 'models/gemini-1.5-flash-001'
safety_settings = [
            {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
generation_config = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}


# PATH OF DATA
DB_FAISS_PATH = 'vectorstore/db_faiss'
CSV_FILE_PATH = 'data/banking_data.csv'