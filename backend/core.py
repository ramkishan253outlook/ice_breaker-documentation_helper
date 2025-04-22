import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain import hub


def run_llm(query, chat_history):
    print("******************** Retrival ******************** ")
    embeddings= OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(embedding=embeddings, index_name=os.environ.get("INDEX_NAME"))
    chat = ChatOpenAI(temperature=0, verbose=True)

    ## Without memory
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    # Without memory
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
        )

    
    retieval_qa_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=combine_docs_chain)

    result =  retieval_qa_chain.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }
    return new_result


if __name__ == "__main__":
    res = run_llm(query="What is a langchain chain?")
    print(res["result"])