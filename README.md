# RAG-Interactive-PDF-Conversational-Chatbot
An interactive question-answering system developed using Langchain, allowing users to upload PDF documents and ask questions contextually based on the document content.

## Features
- **Upload Multiple PDF Files**
- **Contextual Question-Answering**
- **Session Management**
- **Interactive UI with Streamlit**

## Technologies Used
- **Streamlit**: For interactive web application.
- **Groq**: A framework that provides an API for leveraging open source LLMs powered by Language Processing Units (LPUs) to speed up natural language processing.
- **FAISS**: For vector storage and retrieval of document embeddings.

## How to Use
1. Enter your `ChatGroq` API key to initialize the LLM.
2. Upload one or more PDF documents.
3. Enter a question in the chat interface, and the system will provide answers based on the content of the uploaded documents.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/khalil1604/RAG-Interactive-PDF-Conversational-Chatbot
2- cd RAG-Interactive-PDF-Conversational-Chatbot
3- pip install -r requirements.txt
4- streamlit run app.py


  

