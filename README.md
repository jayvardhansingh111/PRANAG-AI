#  PRANAG-AI
#  AI-Powered Crop Specification System

## Overview

Converts natural language agricultural prompts into validated JSON specifications
used by simulation systems to model crop performance under various conditions.

```
User Prompt
    ↓
Orchestrator (AI Brain)     ← LLM + LangGraph + Research APIs
    ↓
Search Engine (Data Layer)  ← Sentence Transformers + ChromaDB
    ↓
Orchestrator (Final JSON)   ← Pydantic validation + spec.json
    ↓
Simulation System           ← Receives validated spec.json
```

---

## Project Structure

```
agri-ai-pipeline/
├── orchestrator/
│   ├── prompt_parser.py        # LLM: raw prompt → structured data
│   ├── workflow.py             # LangGraph: retry + state machine
│   ├── research_fetcher.py     # Semantic Scholar + ArXiv API
│   ├── output_validator.py     # Pydantic strict validation
│   └── spec_builder.py         # Final combiner → spec.json
│
├── search_engine/
│   ├── embeddings.py           # Sentence Transformers: text → vectors
│   ├── vector_store.py         # ChromaDB: store + retrieve 1M+ traits
│   ├── similarity_search.py    # <50ms semantic search API
│   └── data_cleaner.py         # Clean + normalize raw trait data
│
├── shared/
│   ├── models.py               # Pydantic models (shared schema)
│   └── config.py               # Environment variables
│
├── ui/
│   └── app.py                  # Streamlit testing interface
│
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_prompt_parser.py
│   ├── test_output_validator.py
│   ├── test_spec_builder.py
│   ├── test_embeddings.py
│   ├── test_data_cleaner.py
│   ├── test_research_fetcher.py
│   └── test_integration.py
│
├── docs/
│   └── sample_spec.json
├── requirements.txt
└── .env.example
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — add API keys, model paths

# 3. Populate the vector database (first run only)
python search_engine/vector_store.py

# 4. Launch the UI
streamlit run ui/app.py
```

---

## Input / Output Example

**Input prompt:**
```
wheat for Jodhpur at 48°C
```

**Output `spec.json`:**
```json
{
  "crop_type": "wheat",
  "location": { "city": "Jodhpur", "climate_zone": "arid" },
  "conditions": { "temperature_max": 48.0, "stress_type": "heat" },
  "confidence_score": 0.87,
  "validation_passed": true
}
```
