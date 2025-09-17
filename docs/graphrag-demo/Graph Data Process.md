# Graph Data Processing Documentation

## Overview

The Graph Data Processing system extracts entities and relationships from markdown documents using AI language models. The output is a Neo4j-compatible knowledge graph stored as JSON.

## Prerequisites

### System Requirements
- Python 3.11+
- Internet connection for AI API calls
- Minimum 4GB RAM for large document collections

### API Access
Choose one AI provider and obtain API key:
- **Cerebras**: Fast processing, get key at https://cloud.cerebras.ai/
- **Gemini**: Cost-effective, get key at https://aistudio.google.com/

### Dependencies
Install required packages:
```bash
pip install google-generativeai openai orjson json-repair
```

## Configuration

### Environment Variables
Set these variables before running:

```bash
# Required: Choose one AI provider
export CEREBRAS_API_KEY="your_key_here"        # For Cerebras
# OR
export GEMINI_API_KEY="your_key_here"          # For Gemini

# Optional: Processing configuration
export GRAPH_LLM_PROVIDER="cerebras"           # cerebras or gemini
export GRAPH_MIN_ENTITIES="5"                  # Min entities per chunk
export GRAPH_MAX_ENTITIES="15"                 # Max entities per chunk  
export GRAPH_MIN_RELATIONSHIPS="3"             # Min relationships per chunk
export GRAPH_MAX_RELATIONSHIPS="8"             # Max relationships per chunk
export GRAPH_MIN_CONFIDENCE="0.8"              # Quality threshold (0.0-1.0)
export GRAPH_MAX_CONTENT_CHARS="12000"         # Characters per chunk
export GRAPH_SENTENCE_BOUNDARY_RATIO="0.7"    # Chunk boundary detection
```

### Directory Setup
Create the required directory structure:
```
project/
├── workspace/
│   ├── processed/      # Input: Place markdown files here
│   └── graph_data/     # Output: Generated automatically
```

## Usage

### Prerequisites: Document Preparation
Before running graph processing, prepare your documents:

```bash
python 1_crawl_site.py          # Crawl website content
python 2_process_files.py       # Convert to markdown
```

### Step 1: Prepare Input Documents
Place markdown files in `workspace/processed/` directory. Files can be organized in subdirectories.

### Step 2: Run Processing
Execute the graph processing script:
```bash
python 2b_process_graph.py      # Main graph processing
```

### Step 3: Monitor Progress
The system provides real-time progress updates including:
- Files processed vs total
- Entities and relationships extracted
- Processing time estimates
- Error reports

### Directory Structure
```
workspace/
├── processed/          # Input: Markdown files from preparation steps
│   ├── document1.md
│   ├── document2.md
│   └── ...
└── graph_data/        # Output: Generated graph data
    ├── graph_data.json           # Main Neo4j graph
    ├── processed_successfully.txt # Success log
    ├── quota_exceeded_files.txt  # Quota limit log
    ├── failed_files.txt          # Error log
    └── failed_responses.txt      # Raw failed AI responses
```

## Processing Workflow

### Visual Process Flow
```
┌─────────────────┐
│ Start Process   │
└─────────┬───────┘
          │
┌─────────▼───────┐
│ Clean Workspace │
└─────────┬───────┘
          │
┌─────────▼───────┐
│ Load Config &   │
│ Initialize API  │
└─────────┬───────┘
          │
┌─────────▼───────┐
│ Scan for MD     │
│ Files           │
└─────────┬───────┘
          │
┌─────────▼───────┐
│ For Each File:  │
│ ┌─────────────┐ │
│ │ Read Content│ │
│ └─────────────┘ │
│ ┌─────────────┐ │
│ │ Split Chunks│ │
│ └─────────────┘ │
│ ┌─────────────┐ │
│ │ Send to AI  │ │
│ └─────────────┘ │
│ ┌─────────────┐ │
│ │ Extract     │ │
│ │ Entities    │ │
│ └─────────────┘ │
│ ┌─────────────┐ │
│ │ Validate    │ │
│ │ Results     │ │
│ └─────────────┘ │
└─────────┬───────┘
          │
┌─────────▼───────┐
│ Deduplicate     │
│ Entities        │
└─────────┬───────┘
          │
┌─────────▼───────┐
│ Generate Neo4j  │
│ Format          │
└─────────┬───────┘
          │
┌─────────▼───────┐
│ Save Results &  │
│ Generate Reports│
└─────────┬───────┘
          │
┌─────────▼───────┐
│ End Process     │
└─────────────────┘
```

### Detailed Processing Stages

The system follows a structured 7-stage pipeline:

### 1. Environment Initialization
- Validates API credentials and configuration
- Cleans output directory for fresh processing
- Initializes selected AI model connection

### 2. Document Discovery
- Scans input directory recursively for markdown files
- Validates file accessibility and formats
- Generates processing queue with progress tracking

### 3. Content Preparation  
- Normalizes document formatting and whitespace
- Splits large documents into processable chunks
- Maintains context overlap between chunks

### 4. AI-Powered Extraction
- Sends content chunks to AI model for analysis
- Extracts entities with types and confidence scores
- Identifies relationships with supporting evidence
- Implements retry logic for failed extractions

### 5. Quality Validation
- Validates JSON format and required fields
- Filters results below confidence threshold  
- Logs failed extractions for review

### 6. Data Consolidation
- Merges duplicate entities across documents
- Consolidates relationships maintaining highest confidence
- Maintains global entity registry for consistency

### 7. Output Generation
- Converts to Neo4j-compatible format
- Generates comprehensive processing reports
- Saves graph data and audit logs

## Output Structure

### Generated Files
The processing creates the following outputs in `workspace/graph_data/`:

| File | Purpose |
|------|---------|
| `graph_data.json` | Main Neo4j-compatible graph data |
| `processed_successfully.txt` | List of files processed without errors |
| `quota_exceeded_files.txt` | Files that hit API rate limits |  
| `failed_files.txt` | Files that failed with error details |
| `failed_responses.txt` | Raw AI responses that failed validation |

### Graph Data Format

#### Node Structure
```json
{
  "id": "unique-identifier",
  "elementId": "unique-identifier", 
  "labels": ["EntityType"],
  "properties": {
    "name": "Entity Name",
    "content": "Entity description",
    "source": "source_file.md",
    "confidence": 0.95,
    "created_date": "2025-01-15",
    "extraction_method": "cerebras"
  }
}
```

#### Relationship Structure  
```json
{
  "id": "unique-identifier",
  "startNode": "source-node-id",
  "endNode": "target-node-id",
  "type": "RELATIONSHIP_TYPE", 
  "description": "Relationship description",
  "evidence": "Supporting text from document",
  "confidence": 0.90,
  "chunk_id": 0,
  "source_chunk": "chunk_0",
  "source_file": "source_file.md"
}
```

#### Complete File Structure
```json
{
  "nodes": [
    {
      "id": "abc-123",
      "elementId": "abc-123",
      "labels": ["Person"],
      "properties": {
        "name": "John Smith",
        "content": "Software engineer at Tech Corp",
        "source": "company.md",
        "confidence": 0.95,
        "created_date": "2025-01-15",
        "extraction_method": "cerebras"
      }
    }
  ],
  "relationships": [
    {
      "id": "rel-456",
      "startNode": "abc-123",
      "endNode": "def-789",
      "type": "WORKS_AT",
      "description": "Employment relationship",
      "evidence": "John Smith works at Tech Corp",
      "confidence": 0.92,
      "source_file": "company.md"
    }
  ],
  "metadata": {
    "node_count": 150,
    "relationship_count": 89,
    "generated_at": "2025-01-15T10:30:00",
    "generator": "Allycat GraphBuilder",
    "llm_provider": "cerebras",
    "model": "llama-4-scout-17b-16e-instruct",
    "format_version": "neo4j-2025"
  }
}
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRAPH_MIN_ENTITIES` | 5 | Minimum entities extracted per chunk |
| `GRAPH_MAX_ENTITIES` | 15 | Maximum entities extracted per chunk |
| `GRAPH_MIN_RELATIONSHIPS` | 3 | Minimum relationships per chunk |
| `GRAPH_MAX_RELATIONSHIPS` | 8 | Maximum relationships per chunk |
| `GRAPH_MIN_CONFIDENCE` | 0.8 | Quality threshold for inclusion |
| `GRAPH_MAX_CONTENT_CHARS` | 12000 | Maximum characters per processing chunk |

## Error Handling

### Common Issues and Solutions

**API Key Missing**
- Ensure correct environment variable is set
- Verify API key validity with provider

**Quota Exceeded**  
- Processing stops gracefully, saves partial results
- Check API usage limits with provider
- Resume processing after quota resets

**Invalid Document Format**
- Ensure input files are valid markdown
- Check file encoding (UTF-8 recommended)
- Verify file permissions

**Low Quality Extractions**
- Adjust confidence threshold in configuration
- Review failed responses for patterns
- Consider switching AI providers

### Troubleshooting

Enable detailed logging by setting log level to DEBUG in the script. Monitor the generated log files for specific error patterns and processing statistics.

## Performance Considerations

- **Processing Time**: Depends on document count, size, and AI provider
- **API Costs**: Monitor usage with chosen provider
- **Memory Usage**: Large document collections may require additional RAM
- **Network**: Stable internet connection required for AI API calls

The system is designed for batch processing and handles interruptions gracefully by saving progress and providing detailed audit trails.