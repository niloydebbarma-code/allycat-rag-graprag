<img src="assets/allycat.png" alt="Alley Cat" width="200"/>

![License](https://img.shields.io/github/license/The-AI-Alliance/allycat)
![Issues](https://img.shields.io/github/issues/The-AI-Alliance/allycat)
![GitHub stars](https://img.shields.io/github/stars/The-AI-Alliance/allycat?style=social)

# AllyCat

**AllyCat** is full stack, open source chatbot that uses GenAI LLMs to answer questions about your website. It is simple by design and will run on your laptop or server. 

## Why? ##

AllyCat is purposefully simple so it can be used by developers to learn how RAG-based GenAI works. But it is powerful enough to use out of the box, or extend for your own purposes. 

## How does it work? 
AllyCat uses your choice of LLM and vector database to implement a chatbot written in Python using [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) architecture.
AllyCat also includes web scraping tools that extract data from your website (or any website) and import it into AllyCat. 

## 🌟🌟 Features 🌟🌟 

1. Chat app with interface to answers Q&A with data scraped from a website
   - **Default website:** https://thealliance.ai
2. Includes web scraping, extraction and data processing
   - **Current:** Data Prep Kit, Docling
4. Data Prep Kit and Docling. Support for multiple LLMs
5. Support for multiple LLMs
   - **Current:** Llama, Granite
6. Support for multiple vector databases
   - **Current:** Milvus, Weaviate
7. End User and New Contributor Friendly

## Why the name **AllyCat**?

Originally AllianceChat, we shortened it to a cat with a baret when we found out that chat means cat in French. Who doesn't love cats?!

## ⚡️⚡️Quickstart ⚡️⚡️

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
