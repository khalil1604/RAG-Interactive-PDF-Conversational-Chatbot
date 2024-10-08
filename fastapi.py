from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

app = FastAPI()

# Set up embeddings
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize storage for PDF content and chat histories
pdf_storage = {}
chat_histories = {}

# Pydantic models for request and response
class ChatRequest(BaseModel):
    message: str
    session_id: str
    model_name: str

class ChatResponse(BaseModel):
    response: str

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    contents = await file.read()
    pdf_path = f"temp_{file.filename}"
    
    with open(pdf_path, "wb") as f:
        f.write(contents)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    pdf_storage[file.filename] = vectorstore

    os.remove(pdf_path)

    return {"message": f"PDF {file.filename} uploaded and processed successfully"}

@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not pdf_storage:
        raise HTTPException(status_code=400, detail="No PDFs uploaded yet")

    # For simplicity, we'll use the first uploaded PDF's vectorstore
    vectorstore = next(iter(pdf_storage.values()))

    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=request.model_name)

    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str):
        if session not in chat_histories:
            chat_histories[session] = ChatMessageHistory()
        return chat_histories[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, 
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    response = conversational_rag_chain.invoke(
        {"input": request.message},
        config={"configurable": {"session_id": request.session_id}},
    )

    return ChatResponse(response=response['answer'])

