# GraphRAG Demo Setup

## Prerequisites

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create `.env` file with the following settings:

#### Neo4j Configuration
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

#### LLM Configuration
```bash
# Choose one of the following:
LLM_RUN_ENV=cerebras
LLM_MODEL=cerebras/llama3.1-8b

# Or use local Ollama
LLM_RUN_ENV=local_ollama
LLM_MODEL=llama3.1:8b
```

## Pipeline Workflow (Run in Order)

### Step 1: Crawl Website
```bash
python 1_crawl_site.py
```

### Step 2: Process Files
```bash
python 2_process_files.py
```

### Step 3: Save to Vector Database
```bash
python 3_save_to_vector_db.py
```

### Step 4: Process Graph Data
```bash
python 2b_process_graph.py
```

### Step 5: Save to Graph Database
```bash
python 3b_save_to_graph_db.py
```

## Query Applications

### Command Line Query
```bash
# Vector RAG Query
python 4_query.py

# GraphRAG Query
python 4b_query_graph.py
```

### Flask Web Applications
```bash
# Vector RAG Flask (Port 8080)
python app_flask.py

# GraphRAG Flask (Port 8081)
python app_flask_graph.py
```

### Chainlit Chat Applications
```bash
# Vector RAG Chainlit (Port 8000)
chainlit run app_chainlit.py

# GraphRAG Chainlit (Port 8001)
chainlit run app_chainlit_graph.py --port 8001
```

## Application Ports

| Application | Type | Port |
|-------------|------|------|
| Vector RAG Flask | Flask | 8080 |
| GraphRAG Flask | Flask | 8081 |
| Vector RAG Chainlit | Chainlit | 8000 |
| GraphRAG Chainlit | Chainlit | 8001 |