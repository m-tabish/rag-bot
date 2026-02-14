import os
from dotenv import load_dotenv
from generate_result import generate_result
import streamlit as st
from PyPDF2 import PdfReader
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# Constants
CHROMA_PATH = "chroma"
UPLOAD_FOLDER = os.path.abspath("./uploaded_files")

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to process and extract text from a PDF file and return Document objects
def extract_text_from_pdf_to_documents(pdf_path):
    documents = []
    try:
        if not os.path.isfile(pdf_path):
            st.error(f"The file {pdf_path} does not exist.")
            return documents

        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    doc = Document(page_content=text)
                    documents.append(doc)
    except PermissionError:
        st.error(f"Permission denied when trying to access {pdf_path}. Please check your permissions.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return documents

# Main function for generating embeddings and searching results
def main(user_input, documents):
    result = generate_result(user_input, documents)
    return result

# Streamlit app title
st.markdown("""<h1 style="text-align:center; font-family:monospace;">Answer Bot 🤖</h1>""", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; font-size:24px">
        <h5>Upload a PDF and ask questions about its content</h5>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- File Upload Section ---
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

documents = []

if uploaded_file is not None:
    # Save the uploaded file to the uploaded_files folder
    save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ File **{uploaded_file.name}** uploaded successfully!")

    # Extract documents from the uploaded PDF
    documents = extract_text_from_pdf_to_documents(save_path)

    if documents:
        st.info(f"📖 Extracted **{len(documents)}** pages from the PDF.")
    else:
        st.warning("⚠️ Could not extract any text from the uploaded PDF.")

# --- Query Section ---
user_input = st.text_area("What would you like to search?", placeholder="Type your question here...")

if user_input and documents:
    with st.spinner("🔍 Searching for answers..."):
        result = main(user_input, documents)
    
    # Display result in a nice formatted container
    st.markdown("---")
    st.markdown("### 💡 Answer:")
    st.info(result)
elif user_input and not documents:
    st.warning("Please upload a PDF file first.")

# Footer for the app
st.markdown(
    """
    <div style="text-align: center;">
        <h3>Developed by <a href="https://github.com/m-tabish" target="_blank"  display:inline;">Tabish</a></h3>
    </div>
    """,
    unsafe_allow_html=True,
)