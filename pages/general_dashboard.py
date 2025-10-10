from openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables from .env file
load_dotenv(dotenv_path="../medical/.env")

LIGHTRAG_SERVER_URL = os.getenv('LIGHTRAG_SERVER_URL', 'http://localhost:9621')

st.title('MedConsult')

# Sidebar: Model selection and parameters
st.sidebar.title('Model Parameters')
model_source = st.sidebar.radio('Choose Model Source:', ['OpenAI Model', 'Local Offline Model', 'LightRAG Server'])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.1, step=0.1)
max_tokens = st.sidebar.slider('Max Tokens', min_value=1, max_value=4096, value=1028)



# 👇 Set your specific folder path here
FOLDER_PATH = "../medical/inputs"

def render_folder(path):
    """Render PDFs as clickable buttons (links) in sidebar"""
    items = sorted(os.listdir(path))
    for item in items:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            with st.sidebar.expander("📂 " + item, expanded=False):
                render_folder(item_path)
        elif item.lower().endswith(".csv"):
            pdf_path = os.path.abspath(item_path)
            # Fake button using HTML <a> tag styled as Streamlit button
            st.sidebar.markdown(
                f"""
                <a href="file://{pdf_path}" target="_blank">
                    <button style="width: 100%; padding: 6px; border-radius: 6px; border: none; background-color: #f63366; color: white; cursor: pointer;">
                        📄 {item}
                    </button>
                </a>
                """,
                unsafe_allow_html=True,
            )

if os.path.exists(FOLDER_PATH):
    st.sidebar.header("📂 Uploaded files")
    render_folder(FOLDER_PATH)
else:
    st.sidebar.error(f"Path does not exist: {FOLDER_PATH}")

# Initialize session variables
SYSTEM_PROMPT = "You are a helpful medical assistant. Always provide clear, accurate, and empathetic responses to user queries."
LIGHTRAG_CHUNKS_PATH = os.path.join(os.path.dirname(__file__), '..', 'rag_storage', 'kv_store_text_chunks.json')

if 'messages' not in st.session_state:
    # Insert system prompt as the first message
    st.session_state['messages'] = [{"role": "system", "content": SYSTEM_PROMPT}]
if 'model_source' not in st.session_state:
    st.session_state['model_source'] = model_source
if 'uploaded_file_content' not in st.session_state:
    st.session_state['uploaded_file_content'] = None
if 'lightrag_doc_id' not in st.session_state:
    st.session_state['lightrag_doc_id'] = None

# Reset chat history and file if model source changes
if st.session_state['model_source'] != model_source:
    st.session_state['model_source'] = model_source
    st.session_state['messages'] = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state['uploaded_file_content'] = None
    st.session_state['lightrag_doc_id'] = None

# File upload next to chat input
col1, col2 = st.columns([2, 0.5])
with col2:
    uploaded_file = st.file_uploader("Upload file", type=["txt", "pdf", "docx", "csv"], label_visibility="collapsed")  # Removed 'width' argument
    if uploaded_file is not None:
        file_content = uploaded_file.read()
        try:
            file_content = file_content.decode("utf-8")
        except Exception:
            file_content = str(file_content)
        st.session_state['uploaded_file_content'] = file_content
        # Send file to LightRAG server for indexing if selected
        if model_source == 'LightRAG Server':
            try:
                response = requests.post(f"{LIGHTRAG_SERVER_URL}/documents/text", json={"text": file_content})
                response.raise_for_status()
                doc_id = response.json().get('doc_id')
                st.session_state['lightrag_doc_id'] = doc_id
                st.success(f"File uploaded and indexed on LightRAG server. Document ID: {doc_id}")
            except Exception as e:
                st.error(f"Failed to upload file to LightRAG server: {e}")
        else:
            st.success("File uploaded and ready for context.")

with col1:
    prompt = st.chat_input("Enter your query to GPT")

# Display previous messages
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

def get_recent_lightrag_chunks(n=3):
    try:
        with open(LIGHTRAG_CHUNKS_PATH, 'r') as f:
            chunks = json.load(f)
        if isinstance(chunks, dict):
            chunk_list = list(chunks.values())
        else:
            chunk_list = chunks
        return chunk_list[-n:] if len(chunk_list) >= n else chunk_list
    except Exception as e:
        return [f"Error loading LightRAG chunks: {e}"]

# Chat input and response logic
if prompt:
    # Set memory chunk size based on model type
    if model_source == 'OpenAI Model':
        memory_chunk_count = 10 # More memory for online model
    elif model_source == 'Local Offline Model':
        memory_chunk_count = 1  # Less memory for local model
    else:
        memory_chunk_count = 2  # Default for other models

    recent_chunks = get_recent_lightrag_chunks(n=memory_chunk_count)
    memory_context = "\n".join([str(chunk)[:500] for chunk in recent_chunks])  # Limit each chunk to 500 chars

    if st.session_state['uploaded_file_content']:
        file_context = st.session_state['uploaded_file_content'][:500]  # Limit file context
        user_message = f"[Short Memory]\n{memory_context}\n[File Context]\n{file_context}\n[User Question]\n{prompt}"
    else:
        user_message = f"[Short Memory]\n{memory_context}\n[User Question]\n{prompt}"

    st.session_state['messages'].append({"role": "user", "content": user_message})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        if model_source == 'OpenAI Model':
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            stream = client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {"role": message["role"], "content": message["content"]} for message in st.session_state['messages']
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            response = st.write_stream(stream)
        elif model_source == 'Local Offline Model':
            # LMStudio local API (example: http://localhost:1234/v1/chat/completions)
            try:
                lmstudio_url = os.getenv('LMSTUDIO_URL', 'http://localhost:1234/v1/chat/completions')
                payload = {
                    "model": "medgemma-4b-it",  # Replace with your LMStudio model name
                    "messages": [
                        {"role": message["role"], "content": message["content"]} for message in st.session_state['messages']
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                headers = {"Content-Type": "application/json"}
                r = requests.post(lmstudio_url, json=payload, headers=headers)
                r.raise_for_status()
                response_json = r.json()
                response = response_json['choices'][0]['message']['content']
                st.markdown(response)
            except Exception as e:
                response = f"Error: {e}"
                st.markdown(response)
        elif model_source == 'LightRAG Server':
            # Use OpenAI for completion, context from LightRAG
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            stream = client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {"role": message["role"], "content": message["content"]} for message in st.session_state['messages']
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            response = st.write_stream(stream)
        else:
            response = "No model selected."
            st.markdown(response)
    st.session_state['messages'].append({"role": "assistant", "content": response})

# Optionally, handle message overflow or trimming here