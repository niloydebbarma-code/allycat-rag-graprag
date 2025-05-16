<img src="assets/allycat.png" alt="Alley Cat" width="200"/>

# Chat With Data Scraped from any Website

Allycat is designed to:

- crawl a website (we are using [thealliance.ai](https://thealliance.ai/))
- process the HTML, 
- create embeddings and store in a vector database
- and query them using an LLM utilizing a RAG archtecture.


## Built on Open Source Stack

1. Crawling a website: [Data Prep Kit Connector](https://github.com/data-prep-kit/data-prep-kit/blob/dev/data-connector-lib/doc/overview.md)
2. Processing HTML --> MD:  [Docling](https://github.com/docling-project/docling)
3. Processing MD (chunking, saving to vector db): [llama-index](https://docs.llamaindex.ai/en/stable/)
4. Embedding model: [ibm-granite/granite-embedding-30m-english](https://huggingface.co/ibm-granite/granite-embedding-30m-english)
5. Vector Database: [Milvus](https://milvus.io/)
6. LLM:  Use an open source LLM running locally on [ollama](https://ollama.com/) or on a service like [Replicate](https://replicate.com)

## Workflow

![](assets/rag-website-1.png)

## Getting Started

There are two ways to run Allycat.

### Option 1: Quick start using the Allycat docker image

This is great option if you like to do quick evaluation.  
See [running allycat using docker](docs/running-in-docker.md)

### Option 2: Run natively (for tweaking, developing)

Choose this  option if you like to tweak Allycat to fit your needs. For example, experimenting with embedding models or LLMs.  
See [running Allycat natively](docs/running-natively.md)

## AllyCat Workflow

See [running allycat](docs/running-allycat.md)

## Customizing AllyCat

See [customizing allycat](docs/customizing-allycat.md)


## Deploying AllyCat

See [deployment guide](docs/deploy.md)

## Developing AllyCat

[developing allycat](docs/developing-allycat.md)