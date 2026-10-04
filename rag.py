from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing; check your .env file.")


embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedder)
model = ChatOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model="gemini-3.8-flash",
            temperature=0,
)

prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below. If the context doesn't contain
the answer, say "I don't know." Be concise and quote facts directly.

Context:
{context}

Question: {question}
""")

def answer_question(question: str) -> str:
    chunks = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([c.page_content for c in chunks])
    chain=prompt|model
    return chain.invoke({"context":context, "question":question}).content


print(answer_question("How long do I have to return something I bought?"))

                        
                        
