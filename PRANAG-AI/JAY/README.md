# AgriAI Pipeline — AI-Powered Crop Specification System

## Overview

AgriAI converts natural language agricultural prompts into validated JSON crop
specifications. The pipeline combines:

- LLM prompt parsing
- semantic trait search
- scientific research retrieval
- Pydantic schema validation
- Streamlit testing UI

It is designed to produce machine-readable `spec.json` output for downstream
simulation and decision-support systems.

---

## Key Components

- `orchestrator/`
  - `prompt_parser.py` — parse prompt to structured crop, location, and stress data
  - `workflow.py` — pipeline controller with retry state machine
  - `research_fetcher.py` — Semantic Scholar / ArXiv research integration
  - `spec_builder.py` — build final spec JSON from traits, research, and prompt
  - `output_validator.py` — validate and serialize specification

- `search_engine/`
  - `embeddings.py` — generate semantic vectors with SentenceTransformers
  - `vector_store.py` — manage ChromaDB trait index
  - `similarity_search.py` — semantic search over crop trait vectors
  - `data_cleaner.py` — normalize trait data and remove bad records

- `shared/`
  - `models.py` — shared Pydantic schema definitions
  - `config.py` — environment-driven configuration

- `ui/`
  - `app.py` — Streamlit interface for prompt entry and pipeline debugging

---

## Setup

### 1. Install dependencies

```bash
cd JAY
python -m pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
```

Then edit `.env` with your values.

Important variables:

- `OLLAMA_BASE_URL` / `OLLAMA_MODEL` — local LLM endpoint
- `CHROMA_HOST` / `CHROMA_PORT` — ChromaDB configuration
- `EMBEDDING_MODEL` — embedding model name
- `SEMANTIC_SCHOLAR_KEY` — research API key
- `MAX_RETRIES`, `SEARCH_TOP_K`, `MIN_CONFIDENCE`

### 3. Run the UI

From the repository root:

```bash
cd d:\jayvardhan-space\PRANAG-AI
set PYTHONPATH=%cd%
venv\Scripts\streamlit.exe run JAY\ui\app.py
```

> Running from the repository root with `PYTHONPATH` set ensures the `JAY`
> package imports resolve correctly.

---

## Notes

- The UI uses a fallback parser when Ollama is unavailable.
- The vector store uses ChromaDB; local embedded mode is the default.
- If research APIs are unreachable, the pipeline falls back to mock insights.

---

## Testing

Run project tests from the repository root:

```bash
cd d:\jayvardhan-space\PRANAG-AI
set PYTHONPATH=%cd%
venv\Scripts\python.exe -m pytest JAY\tests -v
```

---

## Example

Prompt:

```text
wheat for Jodhpur at 48°C
```

Resulting spec includes crop type, location, climate stress, trait matches,
research insights, and a confidence score.
