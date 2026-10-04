from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def build_index(pdf_path):
    pages       =PyPDFLoader(pdf_path).load()
    splitter    = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50,)
    chunks      = splitter.split_documents(pages)
    embedder    = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedder)
    return vectorstore