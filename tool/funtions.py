import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain


def add_message(user, message):
    st.session_state.chat_history.append({"user": user, "message": message})
    
def load_file_pdf(file_path : str ):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages
    
def process_data(data , embeddings_model):
    faiss_index = FAISS.from_documents(data, embeddings_model)
    retriever = faiss_index.as_retriever()
    return retriever

def chat_history(retriever , model):
    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    ) 
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain
    
def process_url(url):

    # check url
    if not url.startswith('http'):
        st.warning("Invalid URL. Please enter the full URL, starting with 'http' or 'https'.")
        return None

    try:
        response = requests.get(url)
        response.raise_for_status()  

      
        st.success("Valid URLs are accessible")
        return True

    except requests.exceptions.RequestException as e:
        st.error(f"Error accessing URL: {e}")
        return False
    
def get_url(url):
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    content = " ".join(soup.text.split())
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
    texts = text_splitter.create_documents([content])
    return texts

    
    
    
    