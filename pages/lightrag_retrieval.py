import streamlit as st
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="../medical/.env")
LIGHTRAG_SERVER_URL = os.getenv('LIGHTRAG_SERVER_URL', 'http://localhost:9621')

st.set_page_config(page_title="LightRAG Retrieval", page_icon="🔎")
st.title('🔎 LightRAG Retrieval')
st.info('Ask questions about your uploaded data. Results are retrieved from LightRAG and summarized below.')

with st.expander('Retrieval Parameters', expanded=False):
    kg_top_k = st.number_input('KG Top K', min_value=1, max_value=100, value=40)
    chunk_top_k = st.number_input('Chunk Top K', min_value=1, max_value=100, value=10)
    max_entity_tokens = st.number_input('Max Entity Tokens', min_value=100, max_value=50000, value=10000)
    max_relation_tokens = st.number_input('Max Relation Tokens', min_value=100, max_value=50000, value=10000)
    max_total_tokens = st.number_input('Max Total Tokens', min_value=1000, max_value=64000, value=32000)
    enable_rerank = st.checkbox('Enable Rerank', value=True)
    only_need_context = st.checkbox('Only Need Context', value=False)
    only_need_prompt = st.checkbox('Only Need Prompt', value=False)
    stream_response = st.checkbox('Stream Response', value=True)

# 👇 Set your specific folder path here
FOLDER_PATH = "../Healthcare-Assisstant-Dashboard/inputs"

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

# Chat history
if 'lightrag_chat_history' not in st.session_state:
    st.session_state['lightrag_chat_history'] = []

# Display previous messages
for message in st.session_state['lightrag_chat_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input at bottom
query = st.chat_input('Enter your retrieval query:')
if query:
    st.session_state['lightrag_chat_history'].append({"role": "user", "content": query})
    with st.chat_message('user'):
        st.markdown(query)
    query_payload = {
        "query": query,
        "kg_top_k": kg_top_k,
        "chunk_top_k": chunk_top_k,
        "max_entity_tokens": max_entity_tokens,
        "max_relation_tokens": max_relation_tokens,
        "max_total_tokens": max_total_tokens,
        "enable_rerank": enable_rerank,
        "only_need_context": only_need_context,
        "only_need_prompt": only_need_prompt,
        "stream_response": stream_response
    }
    try:
        response = requests.post(f"{LIGHTRAG_SERVER_URL}/query", json=query_payload, timeout=10)
        result = response.json()
        if 'error' in result:
            answer = f"Server error: {result['error']}"
        elif 'response' in result:
            answer = result['response']
        else:
            answer = result.get('summary', 'No summary available.')
        st.session_state['lightrag_chat_history'].append({"role": "assistant", "content": answer})
        with st.chat_message('assistant'):
            st.markdown(answer)
        if 'references' in result:
            st.markdown('**References:**')
            for ref in result['references']:
                st.write(ref)
    except Exception as e:
        error_msg = f"Server error, please try again later: {e}"
        st.session_state['lightrag_chat_history'].append({"role": "assistant", "content": error_msg})
        with st.chat_message('assistant'):
            st.markdown(error_msg)
