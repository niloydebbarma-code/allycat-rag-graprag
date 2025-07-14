# Running LLMs Using Inference Services

We can use any LLM inference service supported by [LiteLLM](https://docs.litellm.ai/docs)

To access the services, you might need to get an API key.

Below are a few services that we have tested with Allycat.

## Nebius

Get an API key at [studio.nebius.com](https://studio.nebius.com/)

Set the following in the `.env` file

```text
NEBIUS_API_KEY = your_nebius_api_key
LLM_MODEL = 'nebius/Qwen/Qwen3-30B-A3B'
```

## Replicate

Get an API key at [replicate.com](https://replicate.com/)

```text
REPLICATE_API_KEY = your_replicate_api_key
LLM_MODEL = 'replicate/meta/meta-llama-3-8b-instruct'
```