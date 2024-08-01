from setuptools import find_packages , setup

setup(
    name = "Chatbot AI",
    version =  '0.0.1',
    author= 'TienLe',
    author_email='tle38413@gmail.com',
    install_requires = ['langchain', "pypdf", "duckduckgo-search", "langchain-community", "wikipedia", "langchain-cohere", "faiss-cpu", "transformers", "requests", "beautifulsoup4", "langchain-core", "streamlit" ,"langchain-huggingface"],
    packages=find_packages()
)