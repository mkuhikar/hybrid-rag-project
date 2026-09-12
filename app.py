

import io
import pandas as pd
import docx
import streamlit as st
from pptx import Presentation
from streamlit_pdf_viewer import pdf_viewer
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

from utils.document_parsers import parse_uploaded_files, extract_text_from_docx
from rag.pipeline import HybridRAGPipeline
from utils.limit_tracker import (
    load_usage_data,
    check_upload_allowed,
    record_upload,
    check_prompt_allowed,
    record_prompt,
    MAX_DAILY_MB,
    MAX_DAILY_PROMPTS
)
def build_index_with_progress(uploaded_files_dict, chunk_size):
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Reading files
    status_text.text("📄 Reading and parsing documents...")
    progress_bar.progress(25)
    parsed_documents = parse_uploaded_files(uploaded_files_dict)

    # Step 2: Preparing text chunks
    status_text.text("✂️ Splitting content into searchable sections...")
    progress_bar.progress(50)

    # Step 3: Generating search index (Embeddings & BM25)
    status_text.text("🧠 Preparing AI search index...")
    progress_bar.progress(75)
    pipeline = HybridRAGPipeline(parsed_documents, chunk_words=chunk_size)

    # Step 4: Completion
    progress_bar.progress(100)
    status_text.text("✨ Document index ready!")
    
    # Clear status indicators after completion
    progress_bar.empty()
    status_text.empty()
    
    return pipeline

# -----------------------------------------------------------------------------
# 1. Page Configuration & Layout Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuChat AI: Interactive Document Assistant",
    page_icon="🧠",
    layout="wide"
)

if "uploaded_files_dict" not in st.session_state:
    st.session_state.uploaded_files_dict = {}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload your documents on the left and ask me anything about them."}
    ]

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

st.markdown("## 🧠 DocuChat AI: Interactive Document Assistant")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. Sidebar Layout: Upload, File Management & Model Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Document Upload & Settings")
    
    # Global Limits Dashboard Banner
    usage = load_usage_data()
    used_mb = usage["uploaded_bytes"] / (1024 * 1024)
    st.info(
        f"🌐 **Global Daily Demo Limits**\n\n"
        f"• **Upload Quota:** {used_mb:.2f} / {MAX_DAILY_MB} MB\n\n"
        f"• **Prompts Used:** {usage['prompt_count']} / {MAX_DAILY_PROMPTS}"
    )

    st.subheader("1. Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Drag and drop or browse files",
        type=["pdf", "docx", "doc", "pptx", "xlsx"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.uploaded_files_dict:
                file_size = f.size
                
                # Check 1: Is this single file alone bigger than the maximum allowed daily cap?
                if (file_size / (1024 * 1024)) > MAX_DAILY_MB:
                    st.error(f"❌ '{f.name}' ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed daily limit ({MAX_DAILY_MB} MB).")
                    continue

                # Check 2: Does adding this file exceed remaining daily quota?
                allowed, err_msg = check_upload_allowed(file_size)
                if not allowed:
                    st.error(err_msg)
                    continue

                # File is valid: store and record usage
                st.session_state.uploaded_files_dict[f.name] = f
                record_upload(file_size)
                st.session_state.rag_pipeline = None  # Invalidate index when a new file is added

    if st.session_state.uploaded_files_dict:
        st.subheader("Current Files")
        files_to_remove = []
        for name in list(st.session_state.uploaded_files_dict.keys()):
            col_name, col_del = st.columns([0.85, 0.15])
            col_name.caption(f"📄 {name}")
            if col_del.button("❌", key=f"del_{name}"):
                files_to_remove.append(name)
        
        for name in files_to_remove:
            del st.session_state.uploaded_files_dict[name]
            st.session_state.rag_pipeline = None
            st.rerun()

    st.markdown("---")
    st.subheader("Settings")
    model_choice = st.selectbox("Model", ["GPT", "QWEN", "GROQ"])
    chunk_size = st.slider(
        "Chunk Size (words)", 
        min_value=20, 
        max_value=500, 
        value=100,
        step=10,
        on_change=lambda: st.session_state.update(rag_pipeline=None)
    )    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    st.markdown("---")
    st.subheader("Processing Status")
    
    if st.session_state.uploaded_files_dict:
        if st.button("⚡ Build/Rebuild Index", use_container_width=True):
            st.session_state.rag_pipeline = build_index_with_progress(
                st.session_state.uploaded_files_dict, 
                chunk_size
            )
            st.success("✔ Index Ready!")
        elif st.session_state.rag_pipeline is not None:
            st.success("✔ Index Ready")
        else:
            st.warning("⚠️ Index out of date. Click build button above.")
    else:
        st.info("ℹ️ No documents uploaded")

# -----------------------------------------------------------------------------
# 3. Helper Functions for Document Previewing
# -----------------------------------------------------------------------------
def render_pdf(file_obj):
    file_obj.seek(0)
    pdf_viewer(
        input=file_obj.read(),
        width="100%",
        height=600,
        pages_to_render=[],
        render_text=True
    )

def render_docx(file_obj):
    try:
        text = extract_text_from_docx(file_obj)
        st.text_area("Document Content", value=text, height=500)
    except Exception as e:
        st.error(f"Error loading DOCX file: {str(e)}")

def render_excel(file_obj):
    file_obj.seek(0)
    excel_file = pd.ExcelFile(file_obj)
    sheet_names = excel_file.sheet_names
    selected_sheet = st.selectbox("Select Sheet", sheet_names)
    df = pd.read_excel(file_obj, sheet_name=selected_sheet)
    st.dataframe(df, use_container_width=True, height=450)

def render_pptx(file_obj):
    prs = Presentation(file_obj)
    st.caption(f"Total Slides: {len(prs.slides)}")
    slide_num = st.number_input("Slide Number", min_value=1, max_value=len(prs.slides), value=1)
    
    slide = prs.slides[slide_num - 1]
    slide_text = []
    for shape in slide.shapes:
        if hasattr(shape, "text"):
            slide_text.append(shape.text)
            
    st.text_area(f"Slide {slide_num} Text", value="\n".join(slide_text), height=450)



# -----------------------------------------------------------------------------
# 4. Main Panel Split (Viewer | Chat UI)
# -----------------------------------------------------------------------------
col_viewer, col_chat = st.columns([0.48, 0.52], gap="medium")

# --- Column 1: Document Viewer ---
with col_viewer:
    st.subheader("Document Viewer")
    
    if st.session_state.uploaded_files_dict:
        selected_filename = st.selectbox(
            "Select File to View",
            options=list(st.session_state.uploaded_files_dict.keys())
        )
        
        file_obj = st.session_state.uploaded_files_dict[selected_filename]
        file_ext = selected_filename.split(".")[-1].lower()
        file_obj.seek(0)
        
        if file_ext == "pdf":
            render_pdf(file_obj)
        elif file_ext in ["docx", "doc"]:
            render_docx(file_obj)
        elif file_ext == "xlsx":
            render_excel(file_obj)
        elif file_ext == "pptx":
            render_pptx(file_obj)
        else:
            st.warning("Unsupported preview format.")
    else:
        st.info("Upload a document from the sidebar to preview it here.")

# --- Column 2: Chat Assistant ---
with col_chat:
    st.subheader("Chat Assistant")
    
    chat_container = st.container(height=500)
    
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "metrics" in msg:
                    st.caption(
                        f"📊 **Context Relevance:** {msg['metrics']['context_relevance']} | "
                        f"**Faithfulness:** {msg['metrics']['faithfulness']} | "
                        f"**Answer Relevance:** {msg['metrics']['answer_relevance']}",
                        help=(
                            "**RAG Triad Metrics Explanation:**\n\n"
                            "• **Context Relevance:** % of query keywords/tokens found in retrieved document contexts.\n"
                            "• **Faithfulness:** % of claims in the generated response that are strictly entailed by the context.\n"
                            "• **Answer Relevance:** Cosine similarity % between your question and the model's answer."
                        )
                    )

    if prompt := st.chat_input("Ask a question about your documents..."):
        prompt_allowed, prompt_err = check_prompt_allowed()
        
        if not prompt_allowed:
            st.error(prompt_err)
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.write(prompt)

            with chat_container:
                with st.chat_message("assistant"):
                    if not st.session_state.uploaded_files_dict:
                        response_text = "Please upload at least one document so I can answer questions about it."
                        st.write(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                    else:
                        # Auto-build with progress bar if index wasn't built yet
                        if st.session_state.rag_pipeline is None:
                            st.session_state.rag_pipeline = build_index_with_progress(
                                st.session_state.uploaded_files_dict, 
                                chunk_size
                            )

                        # User-friendly search spinner
                        with st.spinner("🔍 Reading documents & generating response..."):
                            record_prompt()
                            
                            rag_result = st.session_state.rag_pipeline.query(
                                query_str=prompt,
                                model_name=model_choice,
                                temperature=temperature
                            )
                            
                            response_text = rag_result["generated_answer"]
                            metrics = rag_result["rag_triad_scores"]
                            
                            st.write(response_text)
                            st.caption(
                                f"📊 **Context Relevance:** {metrics['context_relevance']} | "
                                f"**Faithfulness:** {metrics['faithfulness']} | "
                                f"**Answer Relevance:** {metrics['answer_relevance']}",
                                help=(
                                    "**RAG Triad Metrics Explanation:**\n\n"
                                    "• **Context Relevance:** % of query keywords/tokens found in retrieved document contexts.\n"
                                    "• **Faithfulness:** % of claims in the generated response that are strictly entailed by the context.\n"
                                    "• **Answer Relevance:** Cosine similarity % between your question and the model's answer."
                                )
                            )
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response_text,
                                "metrics": metrics
                            })
                            st.rerun()
        