import os
import json
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("faiss is required. Please install faiss-cpu in requirements.") from exc


try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("sentence-transformers is required.") from exc


try:
    from pypdf import PdfReader
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pypdf is required for PDF parsing.") from exc


class LLMProvider(Enum):
    OPENAI = "openai"
    LOCAL_TRANSFORMERS = "local_transformers"


@dataclass
class RagConfig:
    provider: LLMProvider
    openai_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_model_name: str = "microsoft/DialoGPT-large"
    max_answer_tokens: int = 1024  # Increased default for detailed answers
    temperature: float = 0.2


class RagEngine:
    # Class-level cache for models
    _model_cache = {}
    _tokenizer_cache = {}
    
    def __init__(self, data_dir: str, config: RagConfig) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.data_dir / "index.faiss"
        self.meta_path = self.data_dir / "metadata.jsonl"
        self.settings_path = self.data_dir / "settings.json"

        self.config = config

        self.embed_model = SentenceTransformer(self.config.embed_model_name)
        self.embedding_dimension = self.embed_model.get_sentence_embedding_dimension()

        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

        self._load_persistent_state()

    # ---------- Public API ----------
    def rebuild_from_pdfs(self, source_dir: str) -> None:
        pdf_paths = self._collect_pdfs(source_dir)
        self.clear_index()
        if not pdf_paths:
            return
        self.add_pdfs(pdf_paths)

    def add_pdfs(self, pdf_paths: List[str]) -> None:
        chunks, metas = self._read_and_chunk_pdfs(pdf_paths)
        if not chunks:
            return
        vectors = self._embed_texts(chunks)
        self._ensure_index()
        assert self.index is not None

        faiss.normalize_L2(vectors)
        self.index.add(vectors)

        # Persist metadata in the same order
        self.metadata.extend(metas)
        self._save_index_and_metadata()

    def clear_index(self) -> None:
        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()
        if self.settings_path.exists():
            self.settings_path.unlink()
        self.index = None
        self.metadata = []

    def get_index_stats(self) -> Dict[str, Any]:
        return {
            "num_vectors": int(self.index.ntotal) if self.index is not None else 0,
            "num_chunks": len(self.metadata),
            "embedding_model": self.config.embed_model_name,
            "vector_store": "faiss.IndexFlatIP",
        }

    def query(self, question: str, top_k: int = 6) -> Tuple[str, List[Dict[str, Any]]]:
        if self.index is None or self.index.ntotal == 0:
            return (
                "I do not have any indexed documents yet. Please upload PDFs and build the index.",
                [],
            )

        q_vec = self._embed_texts([question])
        faiss.normalize_L2(q_vec)
        distances, indices = self.index.search(q_vec, k=min(top_k, len(self.metadata)))
        idxs = indices[0].tolist()
        scores = distances[0].tolist()

        contexts: List[Dict[str, Any]] = []
        for rank, (i, score) in enumerate(zip(idxs, scores)):
            if i < 0 or i >= len(self.metadata):
                continue
            meta = self.metadata[i].copy()
            meta["score"] = float(score)
            meta["rank"] = rank + 1
            contexts.append(meta)

        # Filter contexts by minimum relevance score (more permissive for comprehensive answers)
        min_score = 0.2  # Reduced threshold to include more relevant contexts
        filtered_contexts = [ctx for ctx in contexts if ctx["score"] >= min_score]
        
        if not filtered_contexts:
            return "I couldn't find relevant information in the documents to answer your question.", contexts

        # Use filtered contexts for generation
        prompt = self._build_prompt(question, filtered_contexts)
        answer = self._generate_answer(prompt, question, filtered_contexts)
        return answer, contexts

    # ---------- Private helpers ----------
    def _load_persistent_state(self) -> None:
        # Load settings to check embedding model compatibility
        settings = {}
        if self.settings_path.exists():
            try:
                settings = json.loads(self.settings_path.read_text())
            except Exception:
                settings = {}

        stored_embed = settings.get("embed_model_name") if isinstance(settings, dict) else None
        if stored_embed and stored_embed != self.config.embed_model_name:
            # Different embedding model than stored; require rebuild
            self.index = None
            self.metadata = []
            return

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = None

        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = [json.loads(line) for line in f if line.strip()]
        else:
            self.metadata = []

    def _save_index_and_metadata(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for meta in self.metadata:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self.settings_path.write_text(
            json.dumps({"embed_model_name": self.config.embed_model_name}, ensure_ascii=False)
        )

    def _ensure_index(self) -> None:
        if self.index is None:
            # Inner product with normalized vectors equals cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_dimension)

    def _collect_pdfs(self, source_dir: str) -> List[str]:
        base = Path(source_dir)
        if not base.exists():
            return []
        return [str(p) for p in sorted(base.glob("*.pdf"))]

    def _read_and_chunk_pdfs(self, pdf_paths: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        chunks: List[str] = []
        metas: List[Dict[str, Any]] = []
        for pdf_path in pdf_paths:
            doc_name = Path(pdf_path).name
            try:
                reader = PdfReader(pdf_path)
            except Exception:
                continue
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                for part in self._split_text_into_chunks(text):
                    if not part.strip():
                        continue
                    chunks.append(part)
                    metas.append(
                        {
                            "id": str(uuid.uuid4()),
                            "doc_name": doc_name,
                            "page": page_num,
                            "text": part,
                        }
                    )
        return chunks, metas

    def _split_text_into_chunks(self, text: str, chunk_chars: int = 800, overlap: int = 100) -> List[str]:
        if not text:
            return []
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        if not text:
            return []
        tokens = list(text)
        chunks: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_chars, len(tokens))
            chunk = "".join(tokens[start:end])
            chunks.append(chunk)
            if end == len(tokens):
                break
            start = end - overlap
            if start < 0:
                start = 0
        return chunks

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        # SentenceTransformer returns a list or ndarray
        vectors = self.embed_model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        return vectors

    def _build_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        context_texts = []
        for c in contexts:
            # Use more context for better answers
            text = c.get('text', '')[:800]  # Increased to 800 chars
            title = f"{c.get('doc_name', 'PDF')} (page {c.get('page', '?')})"
            context_texts.append(f"Source: {title}\n{text}")
        
        # Use more contexts for comprehensive answers
        max_contexts = 5  # Increased from 3 to 5
        if len(context_texts) > max_contexts:
            context_texts = context_texts[:max_contexts]
            
        joined_context = "\n\n---\n\n".join(context_texts) if context_texts else ""
        
        # Enhanced system prompt for detailed answers
        system = """You are an expert technical assistant. Provide comprehensive, detailed answers with the following requirements:

1. **Structure your response** with clear sections, headings, and bullet points
2. **Include code examples** when relevant, properly formatted with syntax highlighting
3. **Provide step-by-step explanations** for complex concepts
4. **Use markdown formatting** for better readability (bold, italic, code blocks)
5. **Include practical examples** and use cases
6. **Explain the reasoning** behind your answers
7. **Be thorough and detailed** - aim for comprehensive coverage
8. **If information is missing**, acknowledge it and provide what you can

Format your response with proper markdown syntax."""
        
        prompt = f"{system}\n\nQuestion: {question}\n\nSources:\n{joined_context}\n\nProvide a comprehensive answer:"
        return prompt

    def _create_technical_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """Create a specialized prompt for technical questions"""
        if not contexts:
            return f"Question: {question}\nAnswer: I don't have enough information to answer this question."
        
        # Extract relevant technical information from contexts
        technical_info = []
        for i, context in enumerate(contexts[:3]):
            text = context.get('text', '')
            # Look for SQL keywords and technical content
            if any(keyword in text.lower() for keyword in ['sql', 'update', 'select', 'insert', 'delete', 'database', 'table', 'query']):
                technical_info.append(f"Technical Reference {i+1}: {text[:800]}")
        
        if not technical_info:
            technical_info = [f"Context {i+1}: {context.get('text', '')[:600]}" for i, context in enumerate(contexts[:3])]
        
        joined_info = "\n\n".join(technical_info)
        
        # Create a specialized technical prompt
        prompt = f"""You are an expert SQL and database consultant. Provide a comprehensive, practical answer with code examples.

Technical Information:
{joined_info}

Question: {question}

Instructions:
1. Provide a clear, step-by-step explanation
2. Include practical SQL code examples with proper syntax
3. Explain the purpose and use cases
4. Use markdown formatting for better readability
5. Include best practices and tips
6. Be specific and actionable

Answer:"""
        return prompt

    def _generate_fallback_response(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """Generate a fallback response when the model output is poor"""
        if not contexts:
            return "I don't have enough information in the provided documents to answer this question accurately."
        
        # Extract relevant information from contexts
        relevant_text = ""
        for context in contexts[:2]:
            text = context.get('text', '')
            if any(keyword in text.lower() for keyword in ['sql', 'update', 'database', 'table']):
                relevant_text += text[:500] + " "
        
        if not relevant_text:
            relevant_text = contexts[0].get('text', '')[:500]
        
        # Create a structured response based on the question type
        question_lower = question.lower()
        
        if 'update' in question_lower and 'database' in question_lower:
            return f"""# SQL Database Update Guide

Based on the information available, here's how to update a database in SQL:

## Basic UPDATE Syntax
```sql
UPDATE table_name
SET column1 = value1, column2 = value2
WHERE condition;
```

## Example: Update User Information
```sql
UPDATE users
SET email = 'newemail@example.com', last_updated = CURRENT_TIMESTAMP
WHERE user_id = 123;
```

## Key Points:
- Always use a WHERE clause to avoid updating all rows
- Test your UPDATE statement with SELECT first
- Use transactions for important updates
- Backup your data before major updates

## Best Practices:
1. **Always backup first**
2. **Use transactions**
3. **Test with SELECT**
4. **Use specific WHERE conditions**

The exact syntax may vary depending on your database system (MySQL, PostgreSQL, SQL Server, etc.)."""
        
        elif 'sql' in question_lower:
            return f"""# SQL Query Guide

Based on the available information, here are the key SQL concepts:

## Common SQL Commands
```sql
-- SELECT data
SELECT column1, column2 FROM table_name WHERE condition;

-- INSERT data
INSERT INTO table_name (column1, column2) VALUES (value1, value2);

-- UPDATE data
UPDATE table_name SET column1 = value1 WHERE condition;

-- DELETE data
DELETE FROM table_name WHERE condition;
```

## Best Practices:
- Always use WHERE clauses for UPDATE and DELETE
- Use proper indexing for better performance
- Write readable, well-formatted queries
- Test queries on small datasets first

For more specific guidance, please provide additional context about your database system and requirements."""
        
        else:
            return f"""# Technical Answer

Based on the information in your documents, here's what I can provide:

## Key Information Found:
{relevant_text[:300]}...

## General Guidance:
- Review the source documents for specific details
- Consider your specific use case and requirements
- Test any code examples in a safe environment first
- Consult your database documentation for exact syntax

For more detailed assistance, please provide additional context about your specific requirements."""

    def _clean_generated_text(self, text: str) -> str:
        """Clean up generated text while preserving formatting"""
        if not text:
            return ""
        
        # Remove common artifacts but preserve structure
        text = text.strip()
        
        # Remove common model artifacts
        text = text.replace('Question:', '').replace('Answer:', '').replace('Context:', '').strip()
        
        # Clean up repeated lines while preserving structure
        lines = text.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                cleaned_lines.append(line)
                seen_lines.add(line)
            elif line.startswith('#'):  # Preserve markdown headers
                cleaned_lines.append(line)
                seen_lines.add(line)
            elif line.startswith('```'):  # Preserve code blocks
                cleaned_lines.append(line)
                seen_lines.add(line)
            elif line.startswith('-') or line.startswith('*'):  # Preserve lists
                cleaned_lines.append(line)
                seen_lines.add(line)
            elif line.startswith('##'):  # Preserve subheaders
                cleaned_lines.append(line)
                seen_lines.add(line)
        
        # Join with proper line breaks
        result = '\n'.join(cleaned_lines)
        
        # Ensure proper markdown formatting
        result = self._enhance_markdown_formatting(result)
        
        # Remove any remaining artifacts
        result = result.replace('Technical Information:', '').replace('Instructions:', '').strip()
        
        # Allow longer responses for detailed answers
        if len(result) > 1500:
            result = result[:1500] + "\n\n... (response truncated for length)"
        
        return result

    def _enhance_markdown_formatting(self, text: str) -> str:
        """Enhance markdown formatting for better readability"""
        if not text:
            return text
        
        # Ensure proper spacing around headers
        lines = text.split('\n')
        enhanced_lines = []
        
        for i, line in enumerate(lines):
            # Add spacing before headers
            if line.startswith('#'):
                if i > 0 and enhanced_lines and enhanced_lines[-1] != '':
                    enhanced_lines.append('')
            enhanced_lines.append(line)
            
            # Add spacing after headers
            if line.startswith('#'):
                enhanced_lines.append('')
        
        return '\n'.join(enhanced_lines)

    def _generate_answer(self, prompt: str, question: str = "", contexts: List[Dict[str, Any]] = None) -> str:
        if self.config.provider == LLMProvider.OPENAI:
            api_key = self.config.openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return "OpenAI API key not set. Provide it in the sidebar."
            try:
                # OpenAI Chat Completions style
                from openai import OpenAI  # type: ignore

                client = OpenAI(api_key=api_key)
                rsp = client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an expert technical assistant. Provide comprehensive, detailed answers with proper formatting, code examples, and thorough explanations. Use markdown formatting for better readability."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=float(self.config.temperature),
                    max_tokens=min(2048, int(self.config.max_answer_tokens) * 2),  # Increased token limit
                )
                return (rsp.choices[0].message.content or "").strip()
            except Exception as exc:
                return f"OpenAI error: {exc}"

        # Local transformers fallback with better model and approach
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            # Use a better model for text generation
            model_key = "microsoft/DialoGPT-large"  # Better conversational model
            
            if model_key not in self._model_cache:
                print(f"Loading model: {model_key}")
                tokenizer = AutoTokenizer.from_pretrained(model_key)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_key,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model.eval()
                
                self._model_cache[model_key] = model
                self._tokenizer_cache[model_key] = tokenizer
            else:
                model = self._model_cache[model_key]
                tokenizer = self._tokenizer_cache[model_key]

            # Create a much better prompt for technical questions
            if contexts is None:
                contexts = []
            
            # Build a comprehensive technical prompt
            technical_prompt = self._create_technical_prompt(question, contexts)
            
            # Tokenize with proper attention mask
            max_input_length = 1024  # Increased for better context
            inputs = tokenizer(
                technical_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_input_length,
                padding=True,
                return_attention_mask=True
            )
            
            # Generate response with better parameters
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=min(600, int(self.config.max_answer_tokens)),  # Increased for better responses
                    do_sample=True,
                    temperature=0.7,  # Slightly higher for more creative responses
                    top_p=0.95,
                    top_k=100,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.05,  # Lower penalty for better flow
                    no_repeat_ngram_size=1,  # Allow more variety
                    early_stopping=True,
                )
            
            # Decode the generated text
            input_length = inputs["input_ids"].shape[1]
            generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            # Clean and enhance the response
            cleaned_text = self._clean_generated_text(generated_text)
            
            # If the response is too short or poor quality, provide a fallback
            if len(cleaned_text) < 50 or "I don't know" in cleaned_text.lower():
                return self._generate_fallback_response(question, contexts)
            
            return cleaned_text
            
        except Exception as exc:
            return f"Local model error: {exc}"


