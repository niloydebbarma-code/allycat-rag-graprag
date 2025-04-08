<img src="assets/allycat.png" alt="Alley Cat" width="200"/>

# Chat With AI Alliance Website

This example will show how to crawl a website, process the HTML, create embeddings in a vector database, and query them using a RAG archtecture.

## Open Source Stack

1. Crawling a website: [Data Prep Kit Connector](https://github.com/data-prep-kit/data-prep-kit/blob/dev/data-connector-lib/doc/overview.md)
2. Processing HTML --> MD:  [Docling](https://github.com/docling-project/docling)
3. Processing MD (chunking, saving to vector db): [llama-index](https://docs.llamaindex.ai/en/stable/)
4. Embedding model: [ibm-granite/granite-embedding-30m-english](https://huggingface.co/ibm-granite/granite-embedding-30m-english)
5. Vector Database: [Milvus](https://milvus.io/)
6. LLM:  [IBM Granite](https://huggingface.co/ibm-granite) or [Llama]()

## Workflow

![](assets/rag-website-1.png)

## Step-1: Setup Python Env

```bash
conda create -n allycat-1  python=3.11

conda activate  allycat-1
```

Install modules

```bash
pip install -r requirements.txt 
```


## Step-2: Configuration

Inspect configuration here: [my_config.py](my_config.py)

You can set AllyCat to :
- site to crawl
- how many files to download and crawl depth
- embedding model
- LLM to use

## Step-3: Crawl a website

This step will crawl a site and download the HTML files into the `input` directory

[1_crawl_site.ipynb](1_crawl_site.ipynb)

For large websites, it is recommended to run the python script as follows

```bash
python     1_crawl_site.py
```


## Step-4: Process HTML files

We will process the HTML files and extract the text as markdown.  The output will be saved in the`output/2-markdown` directory in markdown format

We have 2 processing options:

1. Docling : [2a_process_html_docling.ipynb](2a_process_html_docling.ipynb)
2. Data Prep Kit: [2b_process_html_dpk.ipynb](2b_process_html_dpk.ipynb)

You can run either one.

## Step-5: Save data into DB

Save the extracted text (markdown) into a vector database (Milvus)

[3_save_to_vector_db.ipynb](3_save_to_vector_db.ipynb)

## Step-6: Query documents

### 6.1 - Setup `.env` file with API Token

For this step, we will be using Replicate API service.  We need a Replicate API token for this step.

Follow these steps:

- Get a **free** account at [replicate](https://replicate.com/home)
- Use this [invite](https://replicate.com/invites/a8717bfe-2f3d-4a52-88ed-1356231cdf03) to add some credit  💰  to your Replicate account!
- Create an API token on Replicate dashboard

Once you have an API token, add it to the project like this:

- Copy the file `env.sample.txt` into `.env`  (note the dot in the beginning of the filename)
- Add your token to `REPLICATE_API_TOKEN` in the .env file.

### 6.2 - Query

Query documents using LLM

[4_query.ipynb](4_query.ipynb)

## 7 - Flask UI

```bash
python app.py
```

Go to url : http://localhost:8080

