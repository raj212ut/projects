## PDF RAG Chat (Streamlit)

Upload PDFs, build a local FAISS index with Sentence-Transformers, and ask questions via Retrieval-Augmented Generation. Choose OpenAI (gpt-4 family) or a local Transformers model (default: `google/flan-t5-base`).

### Features
- PDF upload and automatic chunking per page
- Local vector store (FAISS) with persistent metadata
- RAG querying with source citations
- Switch between OpenAI and local model
- Index management (rebuild/clear) from the UI

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If using OpenAI, set your key:
```bash
export OPENAI_API_KEY=sk-...
```

### Run
```bash
streamlit run app.py
```

### Notes
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` by default. Changing it requires index rebuild.
- Local model is CPU-friendly `google/flan-t5-base` (text2text). You may swap it in `rag_pipeline.RagConfig`.
- Uploaded PDFs are saved under `uploads/`, vector index and metadata under `data/`.


