from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

def build_vectorstore(split_docs, persist_dir="db"):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

def get_rag_chain(persist_dir="db"):
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
