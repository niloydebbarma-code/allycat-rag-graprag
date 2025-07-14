# Running LLMs Locally

We run LLMs locally using Ollama or vLLM

## Ollama

[ollama](https://ollama.com/) is used for running open LLMs locally.

Follow [setup instructions from Ollama site](https://ollama.com/download)

### Ollama Models

Ollama supports wide a variety of models.  Which models you can run locally would depend on your local machine configuration

View [available Ollama models](https://ollama.com/models)

To install models: 

```bash
ollama pull <model_name>
ollama pull gemma3:1b
```

Here are some models worthy of consideration ()

| Model                                                 | Parameters | Memory |
|-------------------------------------------------------|------------|--------|
| [qwen3:0.6b](https://ollama.com/library/qwen3)        | 0.6 B      | 522 MB |
| [qwen3:1.7b](https://ollama.com/library/qwen3)        | 1.7 B      | 1.4 GB |
| [tinyllama](https://ollama.com/library/tinyllama)     |            | 638 MB |
| [llama3.2:1b](https://ollama.com/library/llama3.2:1b) | 1 B        | 1.3 GB |
| [gemma3:1b](https://ollama.com/library/gemma3:1b)     | 1 B        | 815 MB |
| [gemma3:4b](https://ollama.com/library/gemma3:4b)     | 4 B        | 3.3 GB |