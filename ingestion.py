
import os
from dotenv import load_dotenv

load_dotenv()


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():
    print("******************* Ingestion *****************************" )

    print("1. Loading.......")
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest", encoding="utf-8")
    documents = loader.load()
    print(f"No of documents >> {len(documents)}")

    print("2. Splitting.......")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    print(f"No of split_docs >> {len(split_docs)}")

    for doc in split_docs:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})



    print("3. Ingesting.......")
    vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.environ.get("INDEX_NAME"))

    # define batch_size
    batch_size = 100

    # loop through documents in batches
    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i: i+batch_size]
        # vectorstore.add_documents(batch)
        # i // batch_size performs integer division to determine the batch index (0-based), and adding 1 converts it to a 1-based batch number.
        print(f"Uploaded batch {i // batch_size + 1} with {len(batch)} documents") 
        
    print("****Loading to vectorstore done ****")
















if __name__ == "__main__":
    ingest_docs()

