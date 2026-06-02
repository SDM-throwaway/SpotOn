## BIPIA benchmark

### Setup

```bash
git submodule update --init --recursive
```

### Running with real API

1. In `docker-compose.yml`, set your API credentials and model:

```yaml
environment:
  - OPENAI_BASE_URL=https://api.openai.com/v1
  - OPENAI_API_KEY=sk-your-key-here
  - TARGET_MODEL=gpt-4o-mini
  - JUDGE_MODEL=gpt-4o-mini
```

2. In the Dockerfile CMD, remove `--mock` and set your desired `--limit` and `--defenses`:

```dockerfile
CMD ["python", "-u", "bipia_benchmark/benchmark_defense.py", "--limit", "100", "--defenses", "none", "delimiting", "encoding"]
```

3. Run:

```bash
docker compose up --build
```

Any OpenAI-compatible endpoint works (OpenAI, Ollama, vLLM, `chat.science.ru.nl/api`).

Each sample hits the API twice per defense (target + judge). `--limit 0` runs the full benchmark.

Available defenses: `none`, `delimiting`, `randomized_delimiting`, `datamarking`, `randomized_datamarking`, `encoding`, `randomized_encoding`

Results go to `bipia_benchmark/experiment_results/email_<timestamp>/`.
