# Cognitive Architecture Book Ingesting Agent (LangGraph)

A reusable LangGraph-based pipeline that ingests a cognitive architecture book (text, PDF, or DOCX) and produces:
- Markdown documentation of components
- PlantUML component diagram
- Knowledge graph JSON
- Validation report

## How it works

Pipeline stages (agents):
1. Ingestor → splits book into semantic chunks
2. Extractor → identifies cognitive components
3. Translator → normalizes names (snake_case) and preserves Romanian labels
4. Deduper → merges near-duplicate entities across batches
5. Graph Builder → builds a knowledge graph (nodes, edges)
6. Diagram Generator → emits a diagram (PlantUML or DOT/Graphviz)
7. Markdown Generator → emits docstring-style Markdown
8. Validator → checks consistency and emits report

All agent prompts are stored under `system_prompts/`.

## Project structure

- `main.py` — entry point
- `config.py` — configuration (paths, models, chunking)
- `agents/` — agent logic (one function per agent)
- `tools/` — utilities (retriever, file I/O)
- `input/` — place your book and instructions
- `output/` — outputs are written here
- `system_prompts/` — all prompts (kept out of code)
- `requirements.txt` — dependencies

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

For CPU-only environments, the `faiss-cpu` package is included. If you hit build issues, install prebuilt wheels per your platform.

## Usage

Place your files:
- Book at `input/book.txt` (supports .txt, .pdf, .docx)
- Instructions at `input/instructions.txt`

Run the pipeline:

```powershell
python main.py
```

Outputs:
- `output/cognitive_architecture.md`
- `output/architecture.puml`
- `output/knowledge_graph.json`
- `output/validation_report.txt`

## Customization

- Modify `system_prompts/*.txt` to change agent behavior without touching code.
- Tweak chunking and models in `config.py`.
- Add new agents by creating a new module in `agents/` and wiring it in `main.py`.

## Notes

- PDF support requires `pdfplumber`; DOCX support requires `python-docx`.
- Uses `sentence-transformers` + FAISS for retrieval. The FAISS index persists under `output/vector_index/` and will be reused across runs unless you delete it or toggle flags in `.env`.
- Diagrams: by default we produce PlantUML (`architecture.puml`). If you prefer to stay offline and avoid Java/CLI, keep the text output. Optionally, set `diagram_format=dot` in `config.py` and use Graphviz locally, or continue with PlantUML CLI by setting `PLANTUML_CLI` in `.env`.