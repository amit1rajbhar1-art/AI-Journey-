from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing; check your .env file.")

model = ChatOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model="gemini-3.8-flash",
            temperature=0,
)   

prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF assistant. Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I couldn't find that in the document."
After your answer, list the page numbers you used as: Sources: page X, page Y.

Context:
{context}


Question: {question}
""")

def ask(vectordb, question) -> str:
    # Assuming vectordb is a vector store and question is a string
    chunks = vectordb.similarity_search(question, k=3)
    context = "\n\n".join(
        f"[page {c.metadata['page']+1}] {c.page_content}" for c in chunks)
    chain = prompt | model
    return chain.invoke({"context": context, "question": question}).content