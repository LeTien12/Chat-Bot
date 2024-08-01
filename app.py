import streamlit as st
from tool.funtions import load_file_pdf , process_data ,chat_history,add_message , process_url , get_url
from tool.model import model_llm 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

wikipedia = WikipediaAPIWrapper()
search = DuckDuckGoSearchRun()

wikipedia_tool = Tool(
    name='wikipedia',
    func= wikipedia.run,
    description="Useful for when you need to look up a topic, country or person on wikipedia"
)

duckduckgo_tool = Tool(
    name='DuckDuckGo Search',
    func= search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
)
memory = ConversationBufferMemory(k = 4)

tools = [wikipedia_tool , duckduckgo_tool]

model = model_llm()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def main():
    st.set_page_config(
            page_title= 'Chat Bot',
            layout= 'centered'
        )
    st.title("Welcome to super smart chatbot")
    
    text = st.chat_input()
    
    with st.sidebar:
        st.title(":blue[URL here] :sunglasses:")
        url = st.sidebar.text_input(f'URL')
        st.title(":blue[Upload file here] :sunglasses:")
        uploade_file = st.file_uploader("" , type='pdf')
        
    if uploade_file:
        with st.spinner("Prosessing..."):
            
            path_file = r"link file" + uploade_file.name
            data = load_file_pdf(path_file)
            
            retriever = process_data(data , embedding)
            rag_chain = chat_history(retriever , model)
            conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    if url:
        check_url = process_url(url)
        if check_url:
            with st.spinner("Prosessing..."):
                content = get_url(url)
                retriever_url = process_data(content , embedding)
                rag_chain_url = chat_history(retriever_url , model)
                conversational_rag_chain_url = RunnableWithMessageHistory(
                rag_chain_url,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
                )
            
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  
        
    for chat in st.session_state.chat_history:
        if chat['user'] == 'user':
            with st.chat_message('User'):
                st.write(chat['message'])
        elif chat['user'] == 'bot':
            with st.chat_message('assistant'):
                st.write(chat['message'])     

    if text:
        if uploade_file:
            with st.chat_message('User'):
                st.write(text)
                add_message("user", text)

            with st.spinner("Prosessing..."):
                result = conversational_rag_chain.invoke(
                            {"input": text},
                            config={
                                "configurable": {"session_id": 'abc123'}
                            },  # constructs a key "abc123" in `store`.
                        )["answer"]
                with st.chat_message('assistant'):
                    st.write(result)
                    add_message("bot", result)
                    
        elif url:
            with st.chat_message('User'):
                st.write(text)
                add_message("user", text)
                
            with st.spinner("Prosessing..."):
                result = conversational_rag_chain_url.invoke(
                            {"input": text},
                            config={
                                "configurable": {"session_id": 'abc123'}
                            },  # constructs a key "abc123" in `store`.
                        )["answer"]
                with st.chat_message('assistant'):
                    st.write(result)
                    add_message("bot", result)
                    
        else:
            zero_shot_agent = initialize_agent(
                agent="zero-shot-react-description",
                tools=tools,
                llm=model,
                verbose=True,
                max_iterations=3,
                memory = memory,
                handle_parsing_errors=True)
            

            with st.chat_message('User'):
                st.write(text)
                add_message("user", text)
            with st.spinner("Prosessing..."):
                result = zero_shot_agent.run(text)
                with st.chat_message('assistant'):
                    st.write(result)
                    add_message("bot", result)
                    
                    
if __name__ == '__main__':
    main()
            
            
            
        
        

        
            
        
            

    
            
        
        
       
        
    


