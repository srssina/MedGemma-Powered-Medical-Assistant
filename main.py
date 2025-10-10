import streamlit as st

st.set_page_config(page_title="Healthcare Assistant", page_icon="🩺")

st.title("🩺 Healthcare Assistant Dashboard ")
st.markdown("""
Welcome to your AI-powered healthcare assistant,This app provides a user-friendly interface for asking general medical questions, accessing patient data, and assisting healthcare professionals.

**Caution:**  
This tool is for research, educational, and demonstration purposes only.  
It is **not** intended for real medical diagnosis, treatment, or commercial use.  
Do **not** rely on this app for any real-world healthcare decisions.
            

**Requirements:**            
- Ensure you have a valid OpenAI API key if using OpenAI models.(set it as an environment variable `OPENAI_API_KEY`)
  You also need openai key for LightRAG integration if don't have proper gpu acesss for local models. 
                     
- For local models, ensure your LMStudio server (or any local llm service provider) is running and accessible.
  download and set up models like `medgemma-4b-it` in LMStudio.
                      
- For LightRAG integration, ensure your LightRAG server is set up and running. Download the LightRAG repository from [here](https://github.com/HKUDS/LightRAG)  and simply use it by typing in command line: `lightrag-server` in the main directory. 

- Preloaded data in rag storage contains 15 Synthetic generated patinet by [synthea](https://synthetichealth.github.io/synthea/) and it's legal to use them for research and educational purposes.
            You can add your own data by uploading files in the dashboard or modifying the LightRAG storage directly.

**Notes:**
  - for smaller models like `medgemma-4b-it`, use a lower `memory_chunk_count` (e.g., 2) to fit within token limits or use fewer uploaded files.

  - make sure to have sufficient GPU memory for local models or have enough tokens in your OpenAI account for LightRAG integration.
            
  - current embedding for lightrag is set to `text-embedding-nomic-embed-text-v1.5` so make sure to have it on your local llm provider.

  - lmstudio uses openai api format in LightRAG so for llm confirguration you must use openai Binding settings and set the host to your local host port.

  - for better use the example `.env` file provided in the repository.
                                                     
                                                       
""")


