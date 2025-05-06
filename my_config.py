import os 

## Configuration
class MyConfig:
    pass 

MY_CONFIG = MyConfig ()

## Crawl settings
MY_CONFIG.CRAWL_URL_BASE = 'https://thealliance.ai/'
MY_CONFIG.CRAWL_MAX_DOWNLOADS = 500
MY_CONFIG.CRAWL_MAX_DEPTH = 5
MY_CONFIG.CRAWL_MIME_TYPE = 'text/html'

## Directories
MY_CONFIG.INPUT_DIR = "input"
MY_CONFIG.OUTPUT_DIR = "output"
### -------------------------------

# Find embedding models: https://huggingface.co/spaces/mteb/leaderboard

# MY_CONFIG.EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# MY_CONFIG.EMBEDDING_LENGTH = 384

# MY_CONFIG.EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'
# MY_CONFIG.EMBEDDING_LENGTH = 384

MY_CONFIG.EMBEDDING_MODEL = 'ibm-granite/granite-embedding-30m-english'
MY_CONFIG.EMBEDDING_LENGTH = 384

## Chunking
MY_CONFIG.CHUNK_SIZE = 512
MY_CONFIG.CHUNK_OVERLAP = 20


### Milvus config
MY_CONFIG.DB_URI = './rag_website_milvus.db'  # For embedded instance
MY_CONFIG.COLLECTION_NAME = 'pages'

## ---- LLM settings ----
## Choose one: We can do local or cloud LLMs
MY_CONFIG.LLM_RUN_ENV = 'replicate'  # 'replicate' or 'local_ollama'
# MY_CONFIG.LLM_RUN_ENV = 'local_ollama'  # 'replicate' or 'local_ollama'

## -- Local LLM --
## We will use Ollama for running local LLMs
## Ollama: https://ollama.com/
## 1. Install Ollama: https://ollama.com/download
## 2. Install models: https://ollama.com/models

if MY_CONFIG.LLM_RUN_ENV == 'local_ollama':
    ## available Ollama models: https://ollama.com/models
    ## install models: ollama pull <model_name>
    ## e.g. ollama pull gemma3:1b
    
    MY_CONFIG.LLM_MODEL = "qwen3:0.6b"      # 522MB
    # MY_CONFIG.LLM_MODEL = "tinyllama"     # 638MB
    # MY_CONFIG.LLM_MODEL = "gemma3:1b"     # 815MB
    # MY_CONFIG.LLM_MODEL = "llama3.2:1b"   # 1.2GB
    # MY_CONFIG.LLM_MODEL = "gemma3:4b"     # 3.3GB
    # MY_CONFIG.LLM_MODEL = "llama3.2:8b"   # 8.1GB
    # MY_CONFIG.LLM_MODEL = "gemma3:2b"     # 1.5GB
    # MY_CONFIG.LLM_MODEL = "gemma3:4b"     # 3.3GB
    # MY_CONFIG.LLM_MODEL = "gemma3:8b"     # 8.1GB


if MY_CONFIG.LLM_RUN_ENV == 'replicate':
    ## LLM Model for replicate service
    ## available models: https://replicate.com/explore
    MY_CONFIG.LLM_MODEL = "meta/meta-llama-3-8b-instruct"
    # MY_CONFIG.LLM_MODEL = "meta/meta-llama-3-70b-instruct"
    # MY_CONFIG.LLM_MODEL = "ibm-granite/granite-3.1-2b-instruct"
    # MY_CONFIG.LLM_MODEL = "ibm-granite/granite-3.2-8b-instruct"
