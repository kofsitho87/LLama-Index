# Using Vector db in Llama Index

## Introduction to Vector Databases

Vector databases play a crucial role in RAG (Retrieval Augmented Generation) systems by providing efficient storage, management, and indexing of high-dimensional vector data. 

They form the foundation for seamless integration with other components, enabling quick and precise information retrieval. Here are some key points explaining the role and importance of vector databases in RAG:
Efficient Knowledge Retrieval: Vector databases act as the backbone for efficient knowledge retrieval in systems.  
They store and manage high-dimensional data, which is essential for RAG models that rely heavily on retrieval for learning and generating accurate responses

Traditional keyword-based search methods have limitations in finding accurate answers to users' questions.
So new method is needed to better understand what users are asking and to find accurate answers in large amounts of data
 

To address this challenge, LLM and vector embedding are key technology.  
- LLM performs well in generating or extracting related answers to users' questions by deeply grasping patterns in textual data. 
- Vector embedding is a technique for representing a word or sentence, or even the meaning of a document as a whole, in vector form, which allows us to quickly compute similarities between documents.
 
The combination of these two technologies is linked by Retrieval Augmented Generation (RAG). 
It means converting a user's question into a vector and finding the most relevant documents or information in the database, which LLM then generates the most appropriate answer to the user's question.

## Understanding Vector Databases

1. Definition and Purpose
Vector databases are a type of database that store and manage unstructured data, such as text, images, or audio, in vector embeddings (high-dimensional vectors) to make it easier to search and query.  
They are designed to handle complex data types and perform high-speed computations, making them well-suited for tasks involving similarity searches and machine learning tasks

2. How Vector Databases Differ from Traditional Databases
- Data Representation: Traditional databases store data in tables, rows, and columns, while vector databases store data in vectors, which are mathematical representations of data points

- Querying Approach: Traditional databases often rely on exact matching queries based on keys or specific attribute values, while vector databases use similarity-based queries, where the goal is to find vectors that are most similar to a given query vector

- Optimization Techniques: Vector databases employ specialized algorithms for Approximate Nearest Neighbor (ANN) search, which optimize the search process. These algorithms may involve techniques such as hashing, quantization, or graph-based search

- Use Cases: Vector databases are often used in applications involving similarity matching, recommendation systems, image recognition, natural language processing, and other tasks that require vector-based operations


```python
!pip install llama-index
```

## Getting Started with Sample Techcrunch Articles

In this tutorial, we will use the [techcrunch](https://techcrunch.com/) dataset to illustrate how to use the RAG system. The dataset contains 10,000 techcrunch articles, each of which contains 10,000 words. The goal of this tutorial is to learn how to use the RAG system to retrieve the most relevant articles for a given question.

#### Download articles
> TechCrunch Article


```python
!wget -q https://github.com/kairess/toy-datasets/raw/master/techcrunch-articles.zip
!unzip -q techcrunch-articles.zip -d articles
```


```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="articles")
docs = reader.load_data()

print(f"Count of Techcrunch articles: {len(docs)}")
```

    Count of Techcrunch articles: 21


### 1. Simple Vector Store


```python
from llama_index.core import VectorStoreIndex


# 1. Load VectorStoreIndex directly from Documents
index = VectorStoreIndex.from_documents(docs, show_progress=True)
```


    Parsing nodes:   0%|          | 0/21 [00:00<?, ?it/s]



    Generating embeddings:   0%|          | 0/51 [00:00<?, ?it/s]


    INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"


#### Persist techcrunch articles in Simple VectorStore


```python
index.set_index_id("techcrunch_articles")
index.storage_context.persist("./stroage/simple")
```

#### Load articles from Simple VectorStore


```python
from llama_index.core import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage/simple")

# load index
simple_vc_index = load_index_from_storage(storage_context, index_id="techcrunch_articles")
```

    INFO:llama_index.core.indices.loading:Loading indices with ids: ['techcrunch_articles']
    Loading indices with ids: ['techcrunch_articles']


### 2. Chroma: Simplifying Vector Database Operations

Chroma is a vector database that is particularly suited for RAG (Retrieval Augmented Generation) systems due to its focus on simplifying the development of large language model (LLM) applications. Chroma is an open-source embedding database that provides developers with a highly-scalable and efficient solution for storing, searching, and retrieving high-dimensional vectors

It is known for its flexibility, allowing deployment on the cloud or as an on-premise solution, and supports multiple data types and formats, making it suitable for a wide range of applications

When comparing Chroma to other vector databases used in RAG systems, it is important to consider their specific strengths and trade-offs. Chroma excels in its flexibility and scalability, making it a popular choice for audio-based search engines, music recommendations, and other audio-related use cases

On the other hand, Pinecone, another vector database, is known for its simple, intuitive interface and extensive support for high-dimensional vector databases, making it suitable for various use cases, including similarity search, recommendation systems, personalization, and semantic search

In terms of scalability, Chroma and Pinecone both support large volumes of high-dimensional data and efficient search performance

However, Pinecone is a fully-managed service, which means it can't be run locally, while Chroma and other vector databases like Milvus, Weaviate, Faiss, Elasticsearch, and Qdrant can be run locally

When choosing the right vector database for your specific needs, consider factors such as scalability, performance, flexibility, ease of use, reliability, and deployment options
Each vector database has its own strengths and trade-offs, so it's essential to evaluate your objectives and choose a vector database that best meets your requirements.


```python
# install chromadb

!pip install chromadb
```

#### Persist techcrunch articles in Chroma VectorStore


```python
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

""" SAVE TO LOCAL"""
db = chromadb.PersistentClient(path="./storage/chroma")
chroma_collection = db.get_or_create_collection("techcrunch_articles")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

chroma_index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)
```

    INFO:chromadb.telemetry.product.posthog:Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
    Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.



    Parsing nodes:   0%|          | 0/21 [00:00<?, ?it/s]



    Generating embeddings:   0%|          | 0/51 [00:00<?, ?it/s]


    INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"


##### Load articles from Chroma VectorStore


```python
db = chromadb.PersistentClient(path="./storage/chroma")
chroma_collection = db.get_or_create_collection("techcrunch_articles")
chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

chroma_index = VectorStoreIndex.from_vector_store(vector_store=chroma_vector_store)
```

### 3. Faiss: Efficient Similarity Search and Clustering


```python
!pip install llama-index-vector-stores-faiss faiss-cpu
```

    Requirement already satisfied: llama-index-vector-stores-faiss in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (0.1.2)
    Requirement already satisfied: faiss-cpu in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (1.8.0)
    Requirement already satisfied: llama-index-core<0.11.0,>=0.10.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-vector-stores-faiss) (0.10.27)
    Requirement already satisfied: numpy in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from faiss-cpu) (1.26.4)
    Requirement already satisfied: PyYAML>=6.0.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (6.0.1)
    Requirement already satisfied: SQLAlchemy>=1.4.49 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (2.0.29)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.6 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (3.9.3)
    Requirement already satisfied: dataclasses-json in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (0.6.4)
    Requirement already satisfied: deprecated>=1.2.9.3 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.2.14)
    Requirement already satisfied: dirtyjson<2.0.0,>=1.0.8 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.0.8)
    Requirement already satisfied: fsspec>=2023.5.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (2024.2.0)
    Requirement already satisfied: httpx in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (0.25.2)
    Requirement already satisfied: llamaindex-py-client<0.2.0,>=0.1.16 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (0.1.16)
    Requirement already satisfied: nest-asyncio<2.0.0,>=1.5.8 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.6.0)
    Requirement already satisfied: networkx>=3.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (3.2.1)
    Requirement already satisfied: nltk<4.0.0,>=3.8.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (3.8.1)
    Requirement already satisfied: openai>=1.1.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.14.2)
    Requirement already satisfied: pandas in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (2.2.1)
    Requirement already satisfied: pillow>=9.0.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (10.2.0)
    Requirement already satisfied: requests>=2.31.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (2.31.0)
    Requirement already satisfied: tenacity<9.0.0,>=8.2.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (8.2.3)
    Requirement already satisfied: tiktoken>=0.3.3 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (0.6.0)
    Requirement already satisfied: tqdm<5.0.0,>=4.66.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (4.66.2)
    Requirement already satisfied: typing-extensions>=4.5.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (4.9.0)
    Requirement already satisfied: typing-inspect>=0.8.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (0.9.0)
    Requirement already satisfied: wrapt in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.16.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (23.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.9.4)
    Requirement already satisfied: pydantic>=1.10 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llamaindex-py-client<0.2.0,>=0.1.16->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (2.6.4)
    Requirement already satisfied: anyio in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (4.3.0)
    Requirement already satisfied: certifi in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (2024.2.2)
    Requirement already satisfied: httpcore==1.* in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.0.4)
    Requirement already satisfied: idna in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (3.6)
    Requirement already satisfied: sniffio in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.3.1)
    Requirement already satisfied: h11<0.15,>=0.13 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from httpcore==1.*->httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (0.14.0)
    Requirement already satisfied: click in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (8.1.7)
    Requirement already satisfied: joblib in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.3.2)
    Requirement already satisfied: regex>=2021.8.3 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (2023.12.25)
    Requirement already satisfied: distro<2,>=1.7.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.9.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from requests>=2.31.0->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from requests>=2.31.0->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (2.2.1)
    Requirement already satisfied: greenlet!=0.4.17 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (3.0.3)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from typing-inspect>=0.8.0->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.0.0)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from dataclasses-json->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (3.20.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (2024.1)
    Requirement already satisfied: packaging>=17.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from marshmallow<4.0.0,>=3.18.0->dataclasses-json->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (23.2)
    Requirement already satisfied: annotated-types>=0.4.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from pydantic>=1.10->llamaindex-py-client<0.2.0,>=0.1.16->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (0.6.0)
    Requirement already satisfied: pydantic-core==2.16.3 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from pydantic>=1.10->llamaindex-py-client<0.2.0,>=0.1.16->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (2.16.3)
    Requirement already satisfied: six>=1.5 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-faiss) (1.16.0)



```python
import faiss

# dimensions of text-ada-embedding-002 in OpenAIEmbedding
d = 1536
faiss_index = faiss.IndexFlatL2(d)
```

    INFO:faiss.loader:Loading faiss.
    Loading faiss.
    INFO:faiss.loader:Successfully loaded faiss.
    Successfully loaded faiss.



```python
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore

faiss_vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=faiss_vector_store)

faiss_index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)

# Save Into Faiss Vector Store
faiss_index.storage_context.persist(persist_dir="./storage/faiss")

```


    Parsing nodes:   0%|          | 0/21 [00:00<?, ?it/s]



    Generating embeddings:   0%|          | 0/51 [00:00<?, ?it/s]


    INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"



```python
# load faiss index from disk
vector_store = FaissVectorStore.from_persist_dir("./storage/faiss")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage/faiss"
)
faiss_index = load_index_from_storage(storage_context=storage_context)
```

    INFO:root:Loading llama_index.vector_stores.faiss.base from ./storage/faiss/default__vector_store.json.
    Loading llama_index.vector_stores.faiss.base from ./storage/faiss/default__vector_store.json.
    INFO:llama_index.core.indices.loading:Loading all indices.
    Loading all indices.


### 4. Qdrant: A Comprehensive Vector Database


```python
!pip install llama-index-vector-stores-qdrant qdrant_client
```

    Requirement already satisfied: llama-index-vector-stores-qdrant in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (0.1.5)
    Requirement already satisfied: grpcio<2.0.0,>=1.60.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-vector-stores-qdrant) (1.62.1)
    Requirement already satisfied: llama-index-core<0.11.0,>=0.10.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-vector-stores-qdrant) (0.10.27)
    Requirement already satisfied: qdrant-client<2.0.0,>=1.7.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-vector-stores-qdrant) (1.8.2)
    Requirement already satisfied: PyYAML>=6.0.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (6.0.1)
    Requirement already satisfied: SQLAlchemy>=1.4.49 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (2.0.29)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.6 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (3.9.3)
    Requirement already satisfied: dataclasses-json in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (0.6.4)
    Requirement already satisfied: deprecated>=1.2.9.3 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.2.14)
    Requirement already satisfied: dirtyjson<2.0.0,>=1.0.8 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.0.8)
    Requirement already satisfied: fsspec>=2023.5.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (2024.2.0)
    Requirement already satisfied: httpx in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (0.25.2)
    Requirement already satisfied: llamaindex-py-client<0.2.0,>=0.1.16 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (0.1.16)
    Requirement already satisfied: nest-asyncio<2.0.0,>=1.5.8 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.6.0)
    Requirement already satisfied: networkx>=3.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (3.2.1)
    Requirement already satisfied: nltk<4.0.0,>=3.8.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (3.8.1)
    Requirement already satisfied: numpy in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.26.4)
    Requirement already satisfied: openai>=1.1.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.14.2)
    Requirement already satisfied: pandas in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (2.2.1)
    Requirement already satisfied: pillow>=9.0.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (10.2.0)
    Requirement already satisfied: requests>=2.31.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (2.31.0)
    Requirement already satisfied: tenacity<9.0.0,>=8.2.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (8.2.3)
    Requirement already satisfied: tiktoken>=0.3.3 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (0.6.0)
    Requirement already satisfied: tqdm<5.0.0,>=4.66.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (4.66.2)
    Requirement already satisfied: typing-extensions>=4.5.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (4.9.0)
    Requirement already satisfied: typing-inspect>=0.8.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (0.9.0)
    Requirement already satisfied: wrapt in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.16.0)
    Requirement already satisfied: grpcio-tools>=1.41.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from qdrant-client<2.0.0,>=1.7.1->llama-index-vector-stores-qdrant) (1.62.1)
    Requirement already satisfied: portalocker<3.0.0,>=2.7.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from qdrant-client<2.0.0,>=1.7.1->llama-index-vector-stores-qdrant) (2.8.2)
    Requirement already satisfied: pydantic>=1.10.8 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from qdrant-client<2.0.0,>=1.7.1->llama-index-vector-stores-qdrant) (2.6.4)
    Requirement already satisfied: urllib3<3,>=1.26.14 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from qdrant-client<2.0.0,>=1.7.1->llama-index-vector-stores-qdrant) (2.2.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (23.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.9.4)
    Requirement already satisfied: protobuf<5.0dev,>=4.21.6 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from grpcio-tools>=1.41.0->qdrant-client<2.0.0,>=1.7.1->llama-index-vector-stores-qdrant) (4.25.3)
    Requirement already satisfied: setuptools in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from grpcio-tools>=1.41.0->qdrant-client<2.0.0,>=1.7.1->llama-index-vector-stores-qdrant) (68.2.2)
    Requirement already satisfied: anyio in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (4.3.0)
    Requirement already satisfied: certifi in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (2024.2.2)
    Requirement already satisfied: httpcore==1.* in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.0.4)
    Requirement already satisfied: idna in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (3.6)
    Requirement already satisfied: sniffio in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.3.1)
    Requirement already satisfied: h11<0.15,>=0.13 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from httpcore==1.*->httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (0.14.0)
    Requirement already satisfied: h2<5,>=3 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from httpx[http2]>=0.20.0->qdrant-client<2.0.0,>=1.7.1->llama-index-vector-stores-qdrant) (4.1.0)
    Requirement already satisfied: click in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (8.1.7)
    Requirement already satisfied: joblib in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.3.2)
    Requirement already satisfied: regex>=2021.8.3 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (2023.12.25)
    Requirement already satisfied: distro<2,>=1.7.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.9.0)
    Requirement already satisfied: annotated-types>=0.4.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from pydantic>=1.10.8->qdrant-client<2.0.0,>=1.7.1->llama-index-vector-stores-qdrant) (0.6.0)
    Requirement already satisfied: pydantic-core==2.16.3 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from pydantic>=1.10.8->qdrant-client<2.0.0,>=1.7.1->llama-index-vector-stores-qdrant) (2.16.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from requests>=2.31.0->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (3.3.2)
    Requirement already satisfied: greenlet!=0.4.17 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (3.0.3)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from typing-inspect>=0.8.0->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.0.0)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from dataclasses-json->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (3.20.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (2024.1)
    Requirement already satisfied: hyperframe<7,>=6.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from h2<5,>=3->httpx[http2]>=0.20.0->qdrant-client<2.0.0,>=1.7.1->llama-index-vector-stores-qdrant) (6.0.1)
    Requirement already satisfied: hpack<5,>=4.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from h2<5,>=3->httpx[http2]>=0.20.0->qdrant-client<2.0.0,>=1.7.1->llama-index-vector-stores-qdrant) (4.0.0)
    Requirement already satisfied: packaging>=17.0 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from marshmallow<4.0.0,>=3.18.0->dataclasses-json->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (23.2)
    Requirement already satisfied: six>=1.5 in /Users/heewungsong/anaconda3/envs/visa_chatbot1/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-vector-stores-qdrant) (1.16.0)



```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# This is for Cloud Storage
# client = QdrantClient(
#     "localhost",
#     port="6333",
# )
client = QdrantClient(path="./storage/qdrant")

qdrant_vector_store = QdrantVectorStore(
    client=client, 
    collection_name="techcrunch_articles", 
    enable_hybrid=False, #  whether to enable hybrid search using dense and sparse vectors
)

storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
qdrant_index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
```


```python
# load Qdrant index from disk

client = QdrantClient(path="./storage/qdrant")

qdrant_vector_store = QdrantVectorStore(client=client, collection_name="techcrunch_articles")
qdrant_index = VectorStoreIndex.from_vector_store(vector_store=qdrant_vector_store)
```

### Conclusion

1. Recap of Key Points
2. Future of Vector Databases in Data-Driven Applications

### 1. Recap of Key Points

In this blog content, we have explored the basics of vector databases and their role in data-driven applications. We have discussed the differences between vector databases and traditional databases, focusing on their data representation, querying approach, optimization techniques, and use cases. We have also provided an overview of three popular vector databases: Chroma, Faiss, and Qdrant, including their key features and usage.

### 2. Future of Vector Databases in Data-Driven Applications
Vector databases are becoming increasingly important in the field of data-driven applications, particularly in tasks involving similarity searches and machine learning.   
As the demand for efficient and scalable data management solutions grows, vector databases are expected to play a more significant role in powering large language models, image recognition, and other AI applications.   

In the future, we can expect to see advancements in vector database technologies, such as improved performance, scalability, and adaptability to different data types. Additionally, we may see the development of more user-friendly APIs and toolkits to facilitate the integration of vector databases into various applications.   

Overall, vector databases are a promising technology that is poised to revolutionize the way we handle and analyze complex data. As the field continues to evolve, we can anticipate new applications and use cases that will further expand the potential of vector databases in data-driven applications.
