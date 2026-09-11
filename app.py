from pdf2image import convert_from_bytes
import pandas as pd
from pypdf import PdfReader
import docx
from pptx import Presentation
import io, base64
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

# -----------------------------------------------------------------------------
# 1. Page Configuration & Layout Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuChat AI: Interactive Document Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize Session State Variables
if "uploaded_files_dict" not in st.session_state:
    st.session_state.uploaded_files_dict = {}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload your documents on the left and ask me anything about them."}
    ]

# Header Title
st.markdown("## 🧠 DocuChat AI: Interactive Document Assistant")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. Sidebar Layout: Upload, File Management & Model Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Document Upload & Settings")
    
    st.subheader("1. Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Drag and drop or browse files",
        type=["pdf", "docx", "doc", "pptx", "xlsx"],
        accept_multiple_files=True
    )
    
    # Store files in session state dictionary by name
    if uploaded_files:
        for f in uploaded_files:
            st.session_state.uploaded_files_dict[f.name] = f

    # Display Current Files list with delete buttons
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
            st.rerun()

    st.markdown("---")
    st.subheader("Settings")
    model_choice = st.selectbox("Model", ["GPT-4o", "GPT-3.5-Turbo", "Claude-3.5-Sonnet"])
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

    st.markdown("---")
    st.subheader("Processing Status")
    if st.session_state.uploaded_files_dict:
        st.success("✔ Index Ready")
    else:
        st.info("ℹ️ No documents uploaded")

# -----------------------------------------------------------------------------
# 3. Helper Functions for Document Previewing
# -----------------------------------------------------------------------------



def render_pdf(file_obj):
    binary_data = file_obj.read()
    
    # Render with responsive scaling and zero extra padding
    pdf_viewer(
        input=binary_data,
        width="100%",           # Fits automatically to column width
        height=600,             # Fixed height with inner scrolling
        pages_to_render=[],     # Render all pages
        render_text=True        # Ensures sharp text rendering
    )

def render_docx(file_obj):
    doc = docx.Document(file_obj)
    full_text = [p.text for p in doc.paragraphs if p.text]
    st.text_area("Document Content", value="\n\n".join(full_text), height=500)

def render_excel(file_obj):
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
        
        # Reset byte stream position
        file_obj.seek(0)
        
        # Render appropriate preview based on extension
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
    
    # Container for rendering historical chat messages
    chat_container = st.container(height=500)
    
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Chat input box at the bottom
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user prompt to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Immediately display user message in panel
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)

        # Generate response (Placeholder logic to connect your backend/LLM)
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if not st.session_state.uploaded_files_dict:
                        response_text = "Please upload at least one document so I can answer questions about it."
                    else:
                        files_list = ", ".join(st.session_state.uploaded_files_dict.keys())
                        response_text = f"I am searching through {files_list} using `{model_choice}` to answer your question:\n\n*\"{prompt}\"*"
                    
                    st.write(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})