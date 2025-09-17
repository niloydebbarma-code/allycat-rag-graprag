import os
from my_config import MY_CONFIG

# If connection to https://huggingface.co/ failed, uncomment the following path
os.environ['HF_ENDPOINT'] = MY_CONFIG.HF_ENDPOINT

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
from llama_index.llms.litellm import LiteLLM
import query_utils
import time
import logging
import json

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_query(query: str):
    global query_engine
    logger.info (f"-----------------------------------")
    start_time = time.time()
    query = query_utils.tweak_query(query, MY_CONFIG.LLM_MODEL)
    logger.info (f"\nProcessing Query:\n{query}")
    res = query_engine.query(query)
    end_time = time.time()
    logger.info ( "-------"
                 + f"\nResponse:\n{res}" 
                 + f"\n\nTime taken: {(end_time - start_time):.1f} secs"
                 + f"\n\nResponse Metadata:\n{json.dumps(res.metadata, indent=2)}" 
                #  + f"\nSource Nodes: {[node.node_id for node in res.source_nodes]}"
                 )
    logger.info (f"-----------------------------------")
## ======= end : run_query =======

## load env config
load_dotenv()

# Setup embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name = MY_CONFIG.EMBEDDING_MODEL
)
logger.info (f"✅ Using embedding model: {MY_CONFIG.EMBEDDING_MODEL}")

# Connect to Vector RAG only database
vector_store = MilvusVectorStore(
    uri = MY_CONFIG.MILVUS_URI_VECTOR,  # Use dedicated Vector-only database
    dim = MY_CONFIG.EMBEDDING_LENGTH,
    collection_name = MY_CONFIG.COLLECTION_NAME, 
    overwrite=False  # so we load the index from db
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
logger.info (f"✅ Connected to Vector-only Milvus instance: {MY_CONFIG.MILVUS_URI_VECTOR}")

# Load Document Index from DB

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, storage_context=storage_context)
logger.info (f"✅ Loaded Vector-only index from: {MY_CONFIG.MILVUS_URI_VECTOR}")

# Setup LLM
logger.info (f"✅ Using LLM model : {MY_CONFIG.LLM_MODEL}")
Settings.llm = LiteLLM (
        model=MY_CONFIG.LLM_MODEL,
    )
 
query_engine = index.as_query_engine()

# Sample queries
queries = [
    # "What is AI Alliance?",
    # "What are the main focus areas of AI Alliance?",
    # "What are some ai alliance projects?",
    # "What are the upcoming events?", 
    # "How do I join the AI Alliance?",
    # "When was the moon landing?",
]

for query in queries:
    run_query(query)

logger.info (f"-----------------------------------")

while True:
    # Get user input
    user_query = input("\nEnter your question (or 'q' to exit): ")
    
    # Check if user wants to quit
    if user_query.lower() in ['quit', 'exit', 'q']:
        logger.info ("Goodbye!")
        break
    
    # Process the query
    if user_query.strip() == "":
        continue
    
    try:
        run_query(user_query)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        print(f"Error processing query: {e}")
