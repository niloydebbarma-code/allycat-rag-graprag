# Hybrid Query Engine

The Hybrid Query Engine is the core component that powers intelligent question-answering by combining vector search with graph database queries. This system implements a sophisticated 7-step workflow that extracts maximum value from both document content and relationship data to provide comprehensive, contextually rich responses.

## Overview

The hybrid approach addresses limitations of traditional RAG systems by combining:

- **Vector Search**: Retrieves relevant document chunks based on semantic similarity
- **Graph Search**: Discovers entities, relationships, and structured knowledge
- **Entity Extraction**: Identifies key concepts from vector results to guide graph queries
- **Fact Synthesis**: Converts graph data into structured knowledge triplets
- **Intelligent Fusion**: Merges both information sources into coherent responses

This creates a "Vector-First Hybrid" system where vector search provides the primary context while graph data adds relationship depth and factual precision.

## Prerequisites

Before using the hybrid query engine, ensure you have:

1. **Vector Database**: Milvus instance with document embeddings
2. **Graph Database**: Neo4j instance with processed knowledge graph
3. **LLM Access**: Configured language model for synthesis
4. **Dependencies**: LlamaIndex, Neo4j driver, and embedding models

## System Architecture

### Core Components

**HybridQueryEngine Class**
- Manages Neo4j Connection class instance and Milvus vector database connections
- Orchestrates the 7-step hybrid query workflow with precise logging from step 1 through 7
- Implements 3-retry mechanisms for targeted graph search with broader fallback strategies
- Handles entity extraction via specialized text processing and query analysis methods
- Processes fact synthesis through triplet extraction and summarization functions

**Connection Management**
- Neo4j Connection class with connect, disconnect, and execute query methods
- Milvus vector store using hybrid GraphRAG database with embedded database files
- HuggingFace embeddings with IBM Granite model for text processing
- LiteLLM interface integrated with query optimization for prompt enhancement
- Automatic connectivity verification and session management

### 7-Step Query Workflow

The engine follows this precise sequence for every query:

1. **Neo4j Connection Verification**: Confirms graph database connectivity using verification protocols
2. **Vector RAG Retrieval (Primary)**: Retrieves semantically relevant document chunks via query engine
3. **Entity Extraction**: Identifies key concepts from vector results using pattern matching
4. **Targeted Graph RAG Retrieval**: Performs 3-retry graph queries using extracted entities
5. **Fact Triplet Extraction**: Converts graph data into structured Subject-Predicate-Object triplets
6. **Facts Summarization**: Creates coherent statements from extracted triplets
7. **Structured Prompt Building & Final Synthesis**: Merges all sources and generates final response

## Configuration

### Environment Setup

The system reads configuration from your `my_config.py` file. Set these environment variables:

```bash
# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j  
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# LLM Configuration
LLM_RUN_ENV=local_ollama  # Options: local_ollama, nebius, replicate, cerebras
LLM_MODEL=ollama/gemma3:1b  # Default for local_ollama

# Alternative LLM Options:
# For Nebius: LLM_MODEL=nebius/Qwen/Qwen3-30B-A3B
# For Replicate: LLM_MODEL=replicate/meta/meta-llama-3-8b-instruct
# For Cerebras: LLM_MODEL=cerebras/llama3.1-8b

# API Keys (if using cloud providers)
# NEBIUS_API_KEY=your_nebius_api_key
# REPLICATE_API_TOKEN=your_replicate_api_token
# CEREBRAS_API_KEY=your_cerebras_api_key
```

### Configuration Values

The system uses these specific configuration values:

```python
# Vector Database - Uses hybrid GraphRAG Milvus instance
MILVUS_URI_HYBRID_GRAPH = "path/to/hybrid_graph_milvus.db"
COLLECTION_NAME = "pages"

# Embedding Model - IBM Granite by default  
EMBEDDING_MODEL = "ibm-granite/granite-embedding-30m-english"
EMBEDDING_LENGTH = 384

# LLM Configuration
LLM_RUN_ENV = "local_ollama"  # Default: local_ollama (options: local_ollama, nebius, replicate, cerebras)
LLM_MODEL = "ollama/gemma3:1b"  # Default model
```

## Usage Examples

### Command Line Interface

When run directly, the script provides an interactive command-line interface:

```bash
python 4b_query_graph.py
# Enter your question (or 'q' to exit): What partnerships does the company have?
```

### Database Testing

The system automatically tests database content on initialization:

```python
# This runs automatically when module loads
engine = HybridQueryEngine()

if engine.neo4j_conn and engine.neo4j_conn.driver:
    print("🔍 Testing database content...")
    test_result = engine.test_database_content()
    print(f"Database test results: {test_result}")
```

**Test Results Include:**
- `node_count`: Total nodes in graph database
- `labels`: Distinct node labels (first 10)
- `sample_nodes`: Sample nodes with properties (first 10)
- `organization_nodes`: Organization nodes ordered by confidence (first 10)
- `relationship_properties`: Relationship types and property keys (first 10)
- `sample_relationships`: Sample relationships with metadata (first 10)

## Advanced Features

### Entity Extraction Strategies

The system employs multiple entity extraction approaches:

**Pattern-Based Extraction**
- Capitalized words and phrases identification
- Organizational terms recognition (company, group, institute)
- Compound terms with hyphens or underscores
- Quoted phrases and meaningful query words

**Content Analysis**
- Frequent word identification and filtering
- Stop word filtering for relevance
- Length-based relevance scoring
- Context-aware entity prioritization

### Graph Search Optimization

**Targeted Search Strategy**
- Uses extracted entities for precise Neo4j queries
- Implements exact and partial name matching with text operations
- Scores matches with weighted relevance: exact name match, partial name, content match
- Retrieves up to 25 most relevant entities ordered by match score and confidence

**3-Level Retry Mechanisms**
- First attempt: Targeted search with extracted entities
- Second attempt: Broader search with shorter entity terms
- Third attempt: Random sampling of high-confidence entities via broader search strategy
- Each retry includes connection error recovery and session management

### Fact Triplet System

**Structured Knowledge Extraction**
```
Subject → Predicate → Object
"OpenAI" → "is_a" → "AI company"
"GPT-4" → "developed_by" → "OpenAI"
```

**Relationship Patterns**
- Entity type classification (is_a)
- Organizational associations (works_with, part_of)
- Functional relationships (develops, manages, provides)
- Collaborative connections (partners_with, collaborates_with)

### Response Synthesis

**Query Engine Integration**
- Uses LlamaIndex `query_engine.query()` for primary synthesis
- Integrates with `query_utils.tweak_query()` for prompt optimization
- Falls back to direct LLM synthesis if query engine fails
- Validates response quality before returning results

**Multi-Source Structured Prompts**
```
SOURCE 1 - Document Information: [vector results]
SOURCE 2 - Knowledge Graph Facts: [summarized facts]
Instructions: Synthesize both sources, prioritize accuracy, highlight relationships
```

## Performance Monitoring

### Detailed Timing Metrics

The system provides comprehensive performance tracking with precise timing:

```
⏱️ Complete Process Summary:
   - Vector RAG: 2.45s
   - Graph RAG: 1.23s  
   - Fact Extraction: 0.0156s
   - Facts Summarization: 0.0089s
   - Final Synthesis: 3.12s
   - Total time: 7.02s
   - Entities extracted: 12
   - Facts extracted: 25
```

### Vector Search Metadata

The system logs vector search statistics including the number of document chunks retrieved and metadata information when available.

### Process Monitoring

Each workflow step includes detailed progress logging:
- `✅ [1/7] Neo4j connection verified`
- `🔄 [2/7] Starting Vector RAG retrieval (Primary)...`
- `🔄 [3/7] Extracting entities from vector results...`
- `🔄 [4/7] Starting targeted Graph RAG retrieval...`
- `🔄 [5/7] Extracting fact triplets from graph data...`
- `🔄 [6/7] Summarizing facts into coherent statements...`
- `🔄 [7/7] Building structured prompt and synthesizing final answer...`

## Error Handling

### Connection Issues

**Neo4j Connection Failures**
```
❌ Failed to connect to Neo4j: [specific error]
❌ Hybrid RAG failed: Neo4j not connected.
❌ Hybrid RAG failed: Neo4j connection error: [specific error]
```
- Verify connection parameters: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- Check Neo4j service is running and accessible
- Validate network connectivity and firewall settings

**Vector Database Issues**
```
❌ Vector setup error: [specific error]
❌ Hybrid RAG failed: Vector engine not available.
❌ Hybrid RAG failed: Vector search error: [specific error]
```
- Check Milvus connection to `MILVUS_URI_HYBRID_GRAPH`
- Verify embedding model configuration and HuggingFace endpoint
- Confirm vector collection exists and is properly indexed

### Data Quality Issues

**Graph Search Failures**
```
❌ All 3 targeted graph search attempts failed
❌ All graph search attempts failed - GRAPH REQUIRED  
❌ This is a graph-dependent system - cannot proceed without graph data
```
- Verify Neo4j database contains processed graph data
- Check entity extraction is producing valid search terms
- Ensure graph nodes have required properties (name, content, confidence)

**Synthesis Issues**
```
❌ Error in query engine synthesis: [specific error]
❌ Error in LLM synthesis: [specific error]  
⚠️ Query engine synthesis returned empty/short response, using fallback
```
- Check LLM model configuration and API availability
- Verify `query_utils.tweak_query()` integration is working
- Review structured prompt building for content issues

## Troubleshooting

### Common Issues

**No Graph Context Retrieved**
```
❌ ALL RETRY STRATEGIES FAILED: Unable to retrieve graph context
❌ This will cause the entire query to fail as graph context is required
```
- Verify Neo4j connection: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` 
- Check that Neo4j database contains processed graph data
- Ensure database name matches `NEO4J_DATABASE` config
- Verify entity extraction is working correctly

**Vector Search Failures**  
```
❌ Vector setup error: [specific error]
```
- Check Milvus connection to `MILVUS_URI_HYBRID_GRAPH`
- Verify collection name matches `COLLECTION_NAME` config  
- Confirm vector indices are built for the hybrid graph collection
- Check embedding model: `EMBEDDING_MODEL` and `EMBEDDING_LENGTH`

**Engine Initialization Issues**
```
Engine not initialized
```
- Verify all required config values in `my_config.py`
- Check environment variables are set correctly
- Ensure both Neo4j and Milvus databases are running
- Review log output for specific initialization errors

### Performance Optimization

**Query Performance Enhancement**
- Monitor individual step timing in system logs
- Consider reducing entity extraction limits for faster processing
- Optimize graph query complexity for better performance
- Use faster embedding models for improved vector search speed

**Memory Usage Management**
- Limit fact triplet extraction counts to manage memory
- Reduce vector search result limits to optimize resource usage
- Implement response caching for repeated queries
- Monitor database session management for efficiency

## Development Notes

### Code Organization

The hybrid query engine is structured as a single standalone module with clear organization:

- **Global Engine Instance**: Engine created at module level after class definition
- **Function Interface**: Run query function provides global access with engine validation
- **Command Line Interface**: Interactive loop when run as main module
- **Automatic Testing**: Database content verification only if Neo4j connection succeeds
- **Comprehensive Logging**: Step-by-step progress tracking through all 7 workflow stages

### Method Structure

**Private Methods (prefixed with `_`)**:
- `_setup_neo4j()`: Initialize Neo4j connection
- `_setup_vector_search()`: Configure Milvus and LlamaIndex components  
- `_extract_entities_from_text()`: Pattern-based entity extraction from vector results
- `_extract_entities_from_query()`: Fallback entity extraction from query text
- `_get_targeted_graph_context()`: 3-retry targeted graph search with broader fallback
- `_get_graph_context_broader_search()`: Final fallback using high-confidence sampling
- `_extract_fact_triplets()`: Convert graph data to Subject-Predicate-Object facts
- `_summarize_facts()`: Group facts by subject and create natural language summaries
- `_build_structured_prompt()`: Combine vector and graph sources into synthesis prompt
- `_synthesize_final_answer()`: Primary synthesis using query engine
- `_fallback_llm_synthesis()`: Direct LLM synthesis when query engine fails

**Public Methods**:
- `test_database_content()`: Returns comprehensive graph database statistics
- `run_query()`: Main 7-step workflow execution
- `close()`: Cleanup Neo4j connections

### Extensibility

The system is designed for easy enhancement:

- **New Entity Patterns**: Add patterns to entity extraction methods for improved recognition
- **Additional Databases**: Extend setup functions for new data sources
- **Custom Synthesis**: Modify synthesis methods for alternative response generation
- **Enhanced Metrics**: Add performance tracking to workflow for better monitoring

### Key Dependencies

- **LlamaIndex**: Core RAG functionality and vector operations
- **Neo4j Python Driver**: Graph database connectivity and operations
- **LiteLLM**: Multi-provider LLM interface for synthesis
- **HuggingFace Transformers**: Embedding model support for text processing
- **Milvus**: Vector similarity search and retrieval

This hybrid approach represents a significant advancement over traditional RAG systems, providing both semantic similarity matching and structured relationship understanding for more comprehensive and accurate responses.