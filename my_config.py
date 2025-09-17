import os 
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

## Configuration

class MyConfig:
    pass

MY_CONFIG = MyConfig ()

## All of these settings can be overridden by .env file
## And it will be loaded automatically by load_dotenv()
## And they will take precedence over the default values below
## See sample .env file 'env.sample.txt' for reference

## HuggingFace config
MY_CONFIG.HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

## Crawl settings
MY_CONFIG.CRAWL_MAX_DOWNLOADS = 100
MY_CONFIG.CRAWL_MAX_DEPTH = 3
MY_CONFIG.WAITTIME_BETWEEN_REQUESTS = int(os.getenv("WAITTIME_BETWEEN_REQUESTS", 0.1)) # in seconds
MY_CONFIG.CRAWL_MIME_TYPE = 'text/html'


## Directories
MY_CONFIG.WORKSPACE_DIR = os.path.join(os.getenv('WORKSPACE_DIR', 'workspace'))
MY_CONFIG.CRAWL_DIR = os.path.join( MY_CONFIG.WORKSPACE_DIR, "crawled")
MY_CONFIG.PROCESSED_DATA_DIR = os.path.join( MY_CONFIG.WORKSPACE_DIR, "processed")

## llama index will download the models to this directory
os.environ["LLAMA_INDEX_CACHE_DIR"] = os.path.join(MY_CONFIG.WORKSPACE_DIR, "llama_index_cache")
### -------------------------------

# Find embedding models: https://huggingface.co/spaces/mteb/leaderboard

MY_CONFIG.EMBEDDING_MODEL =  os.getenv("EMBEDDING_MODEL", 'ibm-granite/granite-embedding-30m-english')
MY_CONFIG.EMBEDDING_LENGTH = int(os.getenv("EMBEDDING_LENGTH", 384))

## Chunking
MY_CONFIG.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
MY_CONFIG.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 20))


### Milvus config
MY_CONFIG.DB_URI = os.path.join( MY_CONFIG.WORKSPACE_DIR, 'rag_website_milvus.db')  # For embedded instance
MY_CONFIG.COLLECTION_NAME = 'pages'

# Separate Milvus databases for different RAG approaches
# This allows running Vector RAG and Hybrid GraphRAG simultaneously without conflicts
MY_CONFIG.MILVUS_URI_VECTOR = os.path.join( MY_CONFIG.WORKSPACE_DIR, 'vector_only_milvus.db')  # Vector RAG only
MY_CONFIG.MILVUS_URI_HYBRID_GRAPH = os.path.join( MY_CONFIG.WORKSPACE_DIR, 'hybrid_graph_milvus.db')  # Hybrid GraphRAG

## ---- LLM settings ----
## Choose one: We can do local or cloud LLMs
# LLM_RUN_ENV controls which LLM backend to use: 'local_ollama' for local Ollama, 'replicate' for Replicate cloud LLMs
# Set LLM_RUN_ENV in your .env file. Default is 'local_ollama' for local development.
## Local LLMs are run on your machine using Ollama
## Cloud LLMs are run on any LiteLLM supported service like Replicate / Nebius / etc
## For running Ollama locally, please check the instructions in the docs/ollama.md file


MY_CONFIG.LLM_RUN_ENV = os.getenv("LLM_RUN_ENV", "local_ollama")


MY_CONFIG.LLM_MODEL = os.getenv("LLM_MODEL", 'ollama/gemma3:1b')

# Replicate API token (if using Replicate)
MY_CONFIG.REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", None)
# Nebius API key (if using Nebius)
MY_CONFIG.NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", None)

# --- GraphBuilder LLM API keys ---
MY_CONFIG.CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", None)
MY_CONFIG.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)

# --- Graph entity/relationship extraction config ---
MY_CONFIG.GRAPH_MIN_ENTITIES = int(os.getenv("GRAPH_MIN_ENTITIES", 5))
MY_CONFIG.GRAPH_MAX_ENTITIES = int(os.getenv("GRAPH_MAX_ENTITIES", 15))
MY_CONFIG.GRAPH_MIN_RELATIONSHIPS = int(os.getenv("GRAPH_MIN_RELATIONSHIPS", 3))
MY_CONFIG.GRAPH_MAX_RELATIONSHIPS = int(os.getenv("GRAPH_MAX_RELATIONSHIPS", 8))
MY_CONFIG.GRAPH_MIN_CONFIDENCE = float(os.getenv("GRAPH_MIN_CONFIDENCE", 0.8))
MY_CONFIG.GRAPH_MAX_CONTENT_CHARS = int(os.getenv("GRAPH_MAX_CONTENT_CHARS", 12000))
MY_CONFIG.GRAPH_SENTENCE_BOUNDARY_RATIO = float(os.getenv("GRAPH_SENTENCE_BOUNDARY_RATIO", 0.7))



## --- GraphRAG ---
# --- Neo4j config ---
MY_CONFIG.NEO4J_URI = os.getenv("NEO4J_URI")
MY_CONFIG.NEO4J_USER = os.getenv("NEO4J_USERNAME")
MY_CONFIG.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
MY_CONFIG.NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
MY_CONFIG.GRAPH_DATA_DIR = os.path.join(MY_CONFIG.WORKSPACE_DIR, "graph_data")





## --- UI settings ---
MY_CONFIG.STARTER_PROMPTS_STR = os.getenv("UI_STARTER_PROMPTS", 'What is this website?  |  What are upcoming events?  | Who are some of the partners?')


MY_CONFIG.STARTER_PROMPTS = MY_CONFIG.STARTER_PROMPTS_STR.split("|") if MY_CONFIG.STARTER_PROMPTS_STR else []