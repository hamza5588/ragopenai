import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import getpass
import os

apikey=os.getenv("apikey")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")


llm = ChatOpenAI(api_key=apikey,model="gpt-3.5-turbo-0125")

embeddings=OpenAIEmbeddings(openai_api_key=apikey)


PINECONE_API_KEY = "38f15772-6ab4-4f8c-b2c6-8b4e2c00f91e"
index_name = "chatai"

vectorstore = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY,index_name=index_name, embedding=embeddings)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Do not provide an answer that is not present in the document, otherwise you will be fined $10,000.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)
custom_rag_prompt = PromptTemplate.from_template(template)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

st.title("Question Answering System")

# Input question from user
question = st.text_input("Enter your question here:")

if st.button("Get Answer"):
    if question:
        # Invoke the RAG model to get the answer
        answer = rag_chain.invoke(question)
        st.markdown(answer)
    else:
        st.warning("Please enter a question.")
