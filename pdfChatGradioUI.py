import gradio as gr
from pdfChatDocIndex import build_index
from pdfChatRetrieval import ask

state = {"db": None}                    # ① remember the index across turns

def upload(pdf):
    state["db"] = build_index(pdf)
    return "✅ PDF indexed! Ask me anything about it."

def chat(message, history):
    history = history or []
    if state["db"] is None:
        answer = "Please upload a PDF first 📄"
    else:
        answer = ask(state["db"], message)
    return history + [[message, answer]], ""

with gr.Blocks(title="📄 Chat with your PDF") as demo:
    gr.Markdown("## 📄 Chat with your PDF (powered by RAG)")
    pdf    = gr.File(label="Upload a PDF", file_types=[".pdf"])
    status = gr.Markdown()
    pdf.upload(upload, inputs=pdf, outputs=status, api_name=False)
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder="Ask a question about your PDF")
    message.submit(
        chat,
        inputs=[message, chatbot],
        outputs=[chatbot, message],
        api_name=False,
    )

if __name__ == "__main__":
    demo.launch(share=True)