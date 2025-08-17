#add open api key in openaiembeddings() , chatopenai()

import os
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from models.supervisor import AgentState, router

# ------------------ Step 1: Load and Split Documents ------------------ #
print("Loading and splitting PDF documents...")

loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# ------------------ Step 2: Embed and Persist Vector DB ------------------ #
if not os.path.exists("db"):
    os.makedirs("db")
print(" Building or loading vector store...")
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory="db")
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# ------------------ Step 3: LLM for RAG & Fallback ------------------ #
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# ------------------ Step 4: RAG Node ------------------ #
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def rag_node(state: AgentState):
    question = state["messages"][-1].content

    prompt = PromptTemplate(
        template="""
You are a helpful medical assistant specialized in osteosarcoma.
Use the retrieved context to answer the user's question.

If the answer is not found in the context, say "I don’t know."

Context:
{context}

Question:
{question}

Answer (max 3 sentences):
""",
        input_variables=["context", "question"]
    )

    chain = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    } | prompt | llm | StrOutputParser()

    response = chain.invoke(question)
    return {"messages": [response]}

# ------------------ Step 5: Fallback LLM Node ------------------ #
def llm_node(state: AgentState):
    question = state["messages"][-1].content
    reply = llm.invoke(f"Answer the question: {question}")
    return {"messages": [reply.content]}

# ------------------ Step 6: Build LangGraph ------------------ #
print("Building agent graph...")

graph = StateGraph(AgentState)
graph.add_node("RAG", rag_node)
graph.add_node("LLM", llm_node)
graph.add_node("Supervisor", lambda state: {"messages": state["messages"]})
graph.set_entry_point("Supervisor")

graph.add_conditional_edges("Supervisor", router, {
    "RAG": "RAG",
    "LLM": "LLM"
})

graph.add_edge("RAG", END)
graph.add_edge("LLM", END)

app = graph.compile()

# ------------------ Step 7: Run App ------------------ #
if __name__ == "__main__":
    print("RAG Agent Ready! Ask me anything. Type 'exit' to quit.\n")
    while True:
        user_input = input("Your question: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        print("\nResponse:")
        print(result["messages"][-1])
