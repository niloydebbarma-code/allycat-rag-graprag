import os
from my_config import MY_CONFIG

# If connection to https://huggingface.co/ failed, uncomment the following path
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.replicate import Replicate
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama


## load env config
load_dotenv()

# Setup embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name = MY_CONFIG.EMBEDDING_MODEL
)
print("✅ Using embedding model: ", MY_CONFIG.EMBEDDING_MODEL)

# Connect to vector db
vector_store = MilvusVectorStore(
    uri = MY_CONFIG.DB_URI,
    dim = MY_CONFIG.EMBEDDING_LENGTH,
    collection_name = MY_CONFIG.COLLECTION_NAME, 
    overwrite=False  # so we load the index from db
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
print("✅ Connected to Milvus instance: ", MY_CONFIG.DB_URI)

# Load Document Index from DB

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, storage_context=storage_context)
print("✅ Loaded index from vector db:", MY_CONFIG.DB_URI)

# Setup LLM
if MY_CONFIG.LLM_RUN_ENV == 'replicate':
    llm = Replicate(
        model=MY_CONFIG.LLM_MODEL,
        temperature=0.1
    )
elif MY_CONFIG.LLM_RUN_ENV == 'local_ollama':
    llm = Ollama(
        model= MY_CONFIG.LLM_MODEL,
        request_timeout=30.0,
        temperature=0.1
    )
else:
    raise ValueError("❌ Invalid LLM run environment. Please set it to 'replicate' or 'local_ollama'.")
print("✅ LLM run environment: ", MY_CONFIG.LLM_RUN_ENV)    
print("✅ Using LLM model : ", MY_CONFIG.LLM_MODEL)
Settings.llm = llm

# Query examples
query_engine = index.as_query_engine()

queries = [
    "What is AI Alliance?",
    "What are the main focus areas of AI Alliance?",
    "What are some ai alliance projects?",
    "What are the upcoming events?", 
    "How do I join the AI Alliance?",
    "When was the moon landing?",
]

for query in queries:
    res = query_engine.query(query)
    print("-----------------------------------")
    print(f"\nQuery: {query}")
    print("------")
    print(f"Response:\n{res}")
    print("-----------------------------------")

print("-----------------------------------")

while True:
    # Get user input
    user_query = input("\nEnter your question (or 'q' to exit): ")
    
    # Check if user wants to quit
    if user_query.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    # Process the query
    try:
        response = query_engine.query(user_query)
        print(f"\nQuery: {user_query}")
        print(f"Response:\n{response}")
        print("-----------------------------------")
    except Exception as e:
        print(f"Error processing query: {e}")
