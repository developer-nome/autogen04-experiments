# import os
# from langchain_community.document_loaders import TextLoader
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter
# from dotenv import load_dotenv
# from autogen_core.models import ModelFamily

# load_dotenv()

# def pretty_print_docs(docs):
#     print(
#         f"\n{'-' * 100}\n".join(
#             [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
#         )
#     )

# documents = TextLoader("state_of_the_union.txt").load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)
# retriever = FAISS.from_documents(texts, OpenAIEmbeddings(
#     api_key=os.getenv("OPEN_AI_API_KEY")
# )).as_retriever()

# docs = retriever.invoke("What did the president say about Ketanji Brown Jackson")
# pretty_print_docs(docs)

#----------------------------------

import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

documents = Docx2txtLoader("quantum-document.docx").load()  #TextLoader("quantum-document.docx").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(texts, OpenAIEmbeddings(
    api_key=os.getenv("OPEN_AI_API_KEY")
)).as_retriever()

llm = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=os.getenv("OPEN_AI_API_KEY"),
        temperature=0
)

# NOTE: DO NOT USE LLMChainFilter.from_llm(llm)
# ^The results are not close

# # NOTE: The LLMChainExtractor compressor was fairly good
# compressor = LLMChainExtractor.from_llm(llm)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=retriever
# )

_filter = LLMListwiseRerank.from_llm(llm, top_n=1)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=retriever
)

# "What did the Qunnect team build to compensate for disturbances of their polarization by vibrations?"
compressed_docs = compression_retriever.invoke(
    "Was Mehdi Namazi involved?"
)
pretty_print_docs(compressed_docs)
