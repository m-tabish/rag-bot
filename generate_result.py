# Generate Result

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# Load environment variables
load_dotenv()

# Constants
CHROMA_PATH = "chroma"
GEMINI_API_KEY = "AIzaSyCeSRDYRwPTmeU4nE9e1aKx7CQyQmVp93Q"
print(GEMINI_API_KEY)
if not GEMINI_API_KEY:
    raise ValueError("Google API key not found. Please set GEMINI_API_KEY in .env file")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context:
{question}
"""

def generate_result(query_text, documents):
 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = text_splitter.split_documents(documents)
 
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
 
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )

    print(f"Stored {len(split_docs)} chunks in ChromaDB")

     
    model = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=GEMINI_API_KEY
    )
 
    results = vector_store.similarity_search(query_text, k=5)

    if not results:
        return "No relevant information found."

    #
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text
    )

     
    response = model.invoke([HumanMessage(content=prompt)])

    # Extract text content properly
    if hasattr(response, 'content'):
        content = response.content
        # If content is a list with dict structure
        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and 'text' in content[0]:
                return content[0]['text']
        # If content is already a string
        elif isinstance(content, str):
            return content
    
    return str(response.content)
