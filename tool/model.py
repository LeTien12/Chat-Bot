from langchain_cohere import ChatCohere

def model_llm():
    model = ChatCohere(model="command-r-plus")
    return model

