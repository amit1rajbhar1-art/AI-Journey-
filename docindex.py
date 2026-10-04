from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


docs = [
    "Our return policy allows refunds within 30 days of purchase.",
    "Shipping is free for orders above ₹999 across India.",
    "For corporate orders above 50 units, contact sales@example.com.",
    "Our office is in Indiranagar, Bangalore. Open Mon-Fri 10am-7pm.",
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=5,
)

chunks = splitter.create_documents(docs)

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(chunks, embedder,persist_directory="./chroma_db")

print(f"Indexed {len(chunks)} chunks 🎉")