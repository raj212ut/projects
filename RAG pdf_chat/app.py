import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import streamlit as st

from rag_pipeline import RagEngine, LLMProvider, RagConfig


APP_TITLE = "PDF RAG Chat"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOAD_REGISTRY_FILE = DATA_DIR / "upload_registry.json"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def load_upload_registry() -> Dict[str, Any]:
    """Load the upload registry from file"""
    if UPLOAD_REGISTRY_FILE.exists():
        try:
            with open(UPLOAD_REGISTRY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"files": {}, "last_updated": None}
    return {"files": {}, "last_updated": None}


def save_upload_registry(registry: Dict[str, Any]) -> None:
    """Save the upload registry to file"""
    with open(UPLOAD_REGISTRY_FILE, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()


def is_file_duplicate(file_content: bytes, registry: Dict[str, Any]) -> Optional[str]:
    """Check if file is a duplicate based on content hash"""
    file_hash = calculate_file_hash(file_content)
    for file_info in registry["files"].values():
        if file_info.get("hash") == file_hash:
            return file_info.get("original_name")
    return None


def get_unique_filename(original_name: str) -> str:
    """Generate a unique filename if the original exists"""
    dest_path = UPLOADS_DIR / original_name
    if not dest_path.exists():
        return original_name
    
    name_no_ext = Path(original_name).stem
    ext = Path(original_name).suffix
    counter = 1
    while dest_path.exists():
        dest_path = UPLOADS_DIR / f"{name_no_ext}__{counter}{ext}"
        counter += 1
    
    return dest_path.name


def add_file_to_registry(file_path: str, original_name: str, file_hash: str) -> None:
    """Add file information to the registry"""
    registry = load_upload_registry()
    
    registry["files"][file_path] = {
        "original_name": original_name,
        "hash": file_hash,
        "upload_date": datetime.now().isoformat(),
        "file_size": Path(file_path).stat().st_size if Path(file_path).exists() else 0
    }
    registry["last_updated"] = datetime.now().isoformat()
    
    save_upload_registry(registry)


def remove_file_from_registry(file_path: str) -> None:
    """Remove file from registry and delete the file"""
    registry = load_upload_registry()
    
    if file_path in registry["files"]:
        del registry["files"][file_path]
        registry["last_updated"] = datetime.now().isoformat()
        save_upload_registry(registry)
    
    # Delete the actual file
    if Path(file_path).exists():
        Path(file_path).unlink()


def get_uploaded_files_info() -> List[Dict[str, Any]]:
    """Get information about all uploaded files"""
    registry = load_upload_registry()
    files_info = []
    
    for file_path, file_info in registry["files"].items():
        if Path(file_path).exists():
            files_info.append({
                "path": file_path,
                "name": file_info["original_name"],
                "upload_date": file_info["upload_date"],
                "file_size": file_info["file_size"],
                "hash": file_info["hash"]
            })
    
    return files_info


def get_default_embed_model() -> str:
    # Good balance of speed and quality
    return "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def get_engine(config: RagConfig) -> RagEngine:
    return RagEngine(data_dir=str(DATA_DIR), config=config)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index_stats" not in st.session_state:
        st.session_state.index_stats = {}


def sidebar_controls() -> Dict[str, Any]:
    st.sidebar.title("Settings")

    provider_label = st.sidebar.selectbox(
        "LLM Provider",
        options=["OpenAI", "Local (Transformers)"],
        index=0,
    )
    provider = LLMProvider.OPENAI if provider_label == "OpenAI" else LLMProvider.LOCAL_TRANSFORMERS

    openai_key: Optional[str] = None
    openai_model = "gpt-4o-mini"
    if provider == LLMProvider.OPENAI:
        openai_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Stored in memory for this session only.",
            value=os.environ.get("OPENAI_API_KEY", ""),
        )
        openai_model = st.sidebar.text_input("OpenAI Model", value=openai_model)
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

    embed_model = st.sidebar.text_input(
        "Embedding Model",
        value=get_default_embed_model(),
        help="Any Sentence-Transformers model ID.",
    )

    top_k = st.sidebar.slider("Context Chunks (k)", min_value=1, max_value=20, value=6, step=1)
    max_tokens = st.sidebar.slider("Max Answer Tokens", min_value=128, max_value=2048, value=512, step=64)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1)

    with st.sidebar.expander("Index Management", expanded=False):
        if st.button("Rebuild Index (from All Files)"):
            # Get all files from registry and rebuild
            files_info = get_uploaded_files_info()
            if not files_info:
                st.warning("No files found in registry.")
            else:
                file_paths = [f['path'] for f in files_info]
                engine = get_engine(
                    RagConfig(
                        provider=provider,
                        openai_model=openai_model,
                        openai_api_key=openai_key,
                        embed_model_name=embed_model,
                        max_answer_tokens=max_tokens,
                        temperature=temperature,
                    )
                )
                with st.spinner(f"Rebuilding index from {len(file_paths)} files..."):
                    engine.rebuild_from_pdfs(source_dir=str(UPLOADS_DIR))
                st.success(f"✅ Index rebuilt from {len(file_paths)} files.")

        if st.button("Clear Index"):
            engine = get_engine(
                RagConfig(
                    provider=provider,
                    openai_model=openai_model,
                    openai_api_key=openai_key,
                    embed_model_name=embed_model,
                    max_answer_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            engine.clear_index()
            st.warning("Index cleared.")

        if st.button("Show Index Stats"):
            engine = get_engine(
                RagConfig(
                    provider=provider,
                    openai_model=openai_model,
                    openai_api_key=openai_key,
                    embed_model_name=embed_model,
                    max_answer_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            st.session_state.index_stats = engine.get_index_stats()

    return {
        "provider": provider,
        "openai_key": openai_key,
        "openai_model": openai_model,
        "embed_model": embed_model,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def upload_section() -> None:
    st.subheader("📁 PDF Management")
    
    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📋 Manage Files", "🔍 File Info"])
    
    with tab1:
        st.write("**Upload New PDFs**")
        uploaded_files = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            registry = load_upload_registry()
            new_files = []
            duplicate_files = []
            
            for file in uploaded_files:
                file_content = file.getbuffer()
                duplicate_name = is_file_duplicate(file_content, registry)
                
                if duplicate_name:
                    duplicate_files.append((file.name, duplicate_name))
                else:
                    new_files.append(file)
            
            # Show duplicate warnings
            if duplicate_files:
                st.warning("⚠️ **Duplicate files detected:**")
                for new_name, existing_name in duplicate_files:
                    st.write(f"• '{new_name}' is identical to '{existing_name}' (already uploaded)")
            
            # Process new files
            if new_files:
                saved_paths: List[str] = []
                for file in new_files:
                    file_content = file.getbuffer()
                    file_hash = calculate_file_hash(file_content)
                    
                    # Get unique filename
                    unique_name = get_unique_filename(file.name)
                    dest_path = UPLOADS_DIR / unique_name
                    
                    # Save file
                    with open(dest_path, "wb") as f:
                        f.write(file_content)
                    
                    # Add to registry
                    add_file_to_registry(str(dest_path), file.name, file_hash)
                    saved_paths.append(str(dest_path))
                
                st.success(f"✅ **Uploaded {len(saved_paths)} new file(s)**")
                
                # Build or update the index with new PDFs
                config = RagConfig(
                    provider=st.session_state.settings["provider"],
                    openai_model=st.session_state.settings["openai_model"],
                    openai_api_key=st.session_state.settings["openai_key"],
                    embed_model_name=st.session_state.settings["embed_model"],
                    max_answer_tokens=st.session_state.settings["max_tokens"],
                    temperature=st.session_state.settings["temperature"],
                )
                engine = get_engine(config)
                with st.spinner("🔄 Building index..."):
                    engine.add_pdfs(saved_paths)
                st.success("✅ **Index updated successfully!**")
    
    with tab2:
        st.write("**Manage Uploaded Files**")
        files_info = get_uploaded_files_info()
        
        if not files_info:
            st.info("📭 No files uploaded yet.")
        else:
            # File management interface
            for i, file_info in enumerate(files_info):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{file_info['name']}**")
                    upload_date = datetime.fromisoformat(file_info['upload_date']).strftime("%Y-%m-%d %H:%M")
                    st.caption(f"Uploaded: {upload_date} | Size: {file_info['file_size']:,} bytes")
                
                with col2:
                    if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                        remove_file_from_registry(file_info['path'])
                        st.rerun()
                
                with col3:
                    if st.button(f"🔄 Re-index", key=f"reindex_{i}"):
                        config = RagConfig(
                            provider=st.session_state.settings["provider"],
                            openai_model=st.session_state.settings["openai_model"],
                            openai_api_key=st.session_state.settings["openai_key"],
                            embed_model_name=st.session_state.settings["embed_model"],
                            max_answer_tokens=st.session_state.settings["max_tokens"],
                            temperature=st.session_state.settings["temperature"],
                        )
                        engine = get_engine(config)
                        with st.spinner(f"Re-indexing {file_info['name']}..."):
                            engine.add_pdfs([file_info['path']])
                        st.success(f"✅ Re-indexed {file_info['name']}")
                        st.rerun()
                
                st.divider()
    
    with tab3:
        st.write("**File Information**")
        files_info = get_uploaded_files_info()
        
        if not files_info:
            st.info("📭 No files uploaded yet.")
        else:
            # Display file statistics
            total_files = len(files_info)
            total_size = sum(f['file_size'] for f in files_info)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", total_files)
            with col2:
                st.metric("Total Size", f"{total_size:,} bytes")
            with col3:
                registry = load_upload_registry()
                last_updated = registry.get("last_updated")
                if last_updated:
                    last_date = datetime.fromisoformat(last_updated).strftime("%Y-%m-%d %H:%M")
                    st.metric("Last Updated", last_date)
            
            # File details table
            st.write("**File Details:**")
            for file_info in files_info:
                with st.expander(f"📄 {file_info['name']}"):
                    st.write(f"**Path:** {file_info['path']}")
                    st.write(f"**Upload Date:** {datetime.fromisoformat(file_info['upload_date']).strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**File Size:** {file_info['file_size']:,} bytes")
                    st.write(f"**Hash:** `{file_info['hash'][:16]}...`")


def chat_section() -> None:
    st.subheader("Ask Questions")

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask something about your PDFs…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        config = RagConfig(
            provider=st.session_state.settings["provider"],
            openai_model=st.session_state.settings["openai_model"],
            openai_api_key=st.session_state.settings["openai_key"],
            embed_model_name=st.session_state.settings["embed_model"],
            max_answer_tokens=st.session_state.settings["max_tokens"],
            temperature=st.session_state.settings["temperature"],
        )
        engine = get_engine(config)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer, contexts = engine.query(
                    question=user_input,
                    top_k=st.session_state.settings["top_k"],
                )
                st.write(answer)

                if contexts:
                    st.caption("Sources")
                    for ctx in contexts:
                        with st.expander(f"{ctx.get('doc_name', 'PDF')} — page {ctx.get('page', '?')} (score: {ctx.get('score', 0):.3f})"):
                            st.write(ctx.get("text", ""))

        st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    ensure_dirs()
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    st.title(APP_TITLE)

    init_session_state()
    settings = sidebar_controls()
    st.session_state.settings = settings

    cols = st.columns(2)
    with cols[0]:
        upload_section()
    with cols[1]:
        chat_section()

    if st.session_state.index_stats:
        st.subheader("Index Stats")
        st.json(st.session_state.index_stats)


if __name__ == "__main__":
    main()


