import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore as LangchainPinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi import UploadFile
from dotenv import load_dotenv
from typing import List


load_dotenv()

class BaseHandler():
    def __init__(self,
                 chat_model:str="meta-llama/llama-4-scout-17b-16e-instruct",
                 temperature:float=0.6,
                 **kwargs):
        self.pinecone_api_key=os.getenv("PINECONE_API_KEY")
        self.pinecone_env=os.getenv("PINECONE_ENV")
        self.pinecone_index=os.getenv("PINECONE_INDEX")
        self.groq_api_key=os.getenv("GROQ_API_KEY")
        self.chat_model=chat_model
        self.embeddings=HuggingFaceEmbeddings()
        self.llm=ChatGroq(model=self.chat_model,
                          api_key=self.groq_api_key,
                          temperature=temperature)
        #print(self.groq_api_key)
        self.pinecone_client=Pinecone(api_key=self.pinecone_api_key)
        print(self.pinecone_index)
        self.index = self.pinecone_client.Index(self.pinecone_index)
        

    
    def load_data(self,file_path:str):
        loader=PyMuPDFLoader(file_path)
        documents=loader.load()
        return documents
        
    def ingest_data(self,documents,chunk_size:int=1000,chunk_overlap:int=100,**kwargs):
        

        text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        result=text_splitter.split_documents(documents)

        

        LangchainPinecone.from_documents(result,
                                embedding=self.embeddings,
                                index_name=self.pinecone_index
                                )
        
    def chat(self,query:str,chat_history:List[str]):
        qa_prompt = ChatPromptTemplate.from_template(
                                    """
                                    You are an assistant for question-answering tasks. 
                                    Use the following pieces of retrieved context to answer the question. 
                                    If you don't know the answer, say that you don't know. 
                                    Use three sentences maximum and keep the answer concise.
                                    
                                    <context>
                                    {context}
                                    </context>
                                    <chat_history>
                                    {chat_history}

                                    Question: {input}
                                    """
                                )
        system_prompt = ChatPromptTemplate.from_template("""
                Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

                Chat History:
                {chat_history}
                Follow Up Question: {input}
                Standalone question:
                """)

        db=LangchainPinecone.from_existing_index(
            index_name=self.pinecone_index, 
                embedding=self.embeddings, 
                text_key='text', 
                
        )
        retriever=db.as_retriever()
        hisory_aware_retriever=create_history_aware_retriever(self.llm,
                                                              retriever,
                                                             system_prompt)
        combine_docs_chain = create_stuff_documents_chain(
                                    llm=self.llm,
                                    prompt=qa_prompt
                            )
        retriever_chain=create_retrieval_chain(hisory_aware_retriever,combine_docs_chain)
        chat_history_str = "\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in chat_history]
                                    )
        result=retriever_chain.invoke({"input":query,"chat_history":chat_history_str})
        return result
