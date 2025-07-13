# Allycat Configuration

Here are all the config parameters for Allycat.

The default config values are in [my_config.py](../my_config.py)

Some config values can be overridden in `.env` file

## Crawl Configurations

| Configuration       | Default Value | Description                                                                                            |
|---------------------|---------------|--------------------------------------------------------------------------------------------------------|
| CRAWL_MAX_DOWNLOADS | 100           | How many files to download.  Override this by specifying `--max-downloads` to script `1_crawl_site.py` |
| CRAWL_MAX_DEPTH     | 3             | How many levels to crawl.  Override this by specifying `--depth` to script `1_crawl_site.py`           |
| WAITTIME_BETWEEN_REQUESTS     | 0.1             | How long to wait before making download request (in seconds). <br> Override in `.env` file       |

## Workspace

This is where the artifacts are saved

| Configuration       | Default Value | Description                                                                                            |
|---------------------|---------------|--------------------------------------------------------------------------------------------------------|
| WORKSPACE_DIR | workspace           | where files / models / databases are stored. <br> Override in `.env` file |

## Embeddings

We support open-source embeddings.  You can find them at [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

| Configuration       | Default Value | Description                                                                                            |
|---------------------|---------------|--------------------------------------------------------------------------------------------------------|
| EMBEDDING_MODEL | `ibm-granite/granite-embedding-30m-english`           | Embedding model to use. <br> Override in `.env` file |
| EMBEDDING_LENGTH | 384           | Embedding vector size. <br> Override in `.env` file. <br>  Must match `EMBEDDING_MODEL` setting above |
| CHUNK_SIZE | 512           | Chunk size  <br> Override in `.env` file. |
| CHUNK_OVERLAP | 20           | Chunk overlap  <br> Override in `.env` file. |

