import streamlit as st
from PIL import Image
import base64
import io
import os
import time
import requests


def load_sample_image():
    sample_path = os.path.join(os.path.dirname(__file__), "../images/Infection.jpg")
    sample_path = os.path.normpath(sample_path)
    if os.path.exists(sample_path):
        return Image.open(sample_path)
    alt = os.path.join(os.getcwd(), "images/Infection.jpg")
    if os.path.exists(alt):
        return Image.open(alt)
    return None

def analyze_image_with_model(pipe, image, custom_prompt: str):
    if image is None:
        return "Please upload an image first."
    system_prompt_text = (
        "You are a expert medical AI assistant with years of experience in interpreting medical images. "
        "Your purpose is to assist qualified clinicians by providing a detailed analysis of the provided medical image."
    )
    prompt_text = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else "Describe this image in detail, including any abnormalities or notable findings."
    # Encode image to JPEG and base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"
    # OpenAI-compatible multimodal format: text and image as separate items
    payload = {
        "model": pipe["model"],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt_text}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    headers = {"Content-Type": "application/json"}
    r = requests.post(pipe["url"], json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    result = r.json()
    try:
        return result["choices"][0]["message"]["content"]
    except Exception:
        return result.get("text") or str(result)

def load_model():
    """Return LM Studio API config with hardcoded endpoint and model."""
    lmstudio_url = "http://localhost:1234/v1/chat/completions"
    lmstudio_model = "medgemma-4b-it"
    return {"type": "lmstudio", "url": lmstudio_url, "model": lmstudio_model}


def main():
    st.markdown("# Medgemma VLM Medical Image Analysis 🧠")

    lmstudio_url = "http://localhost:1234/v1/chat/completions"
    lmstudio_model = "medgemma-4b-it"

    st.info(
        "This model is for educational and research purposes only. It is not a substitute for professional medical diagnosis or treatment."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### \U0001F4E4 Upload Medical Image (Radiology, Pathology, Dermatology, CT, X-Ray)")
        uploaded_file = st.file_uploader("Input Image", type=["png", "jpg", "jpeg", "bmp", "tiff"], accept_multiple_files=False)

        if st.button("\U0001F4CB Load Sample Image"):
            sample = load_sample_image()
            if sample:
                st.session_state["_sample_image"] = sample
            else:
                st.warning("Sample image not found in repo (images/Infection.jpg)")

        custom_prompt = st.text_area(
            "\U0001F4AC Custom Analysis Prompt (Optional)",
            value="Describe this Image and Generate a compact Clinical report",
            height=100,
        )

        analyze = st.button("\U0001F50D Analyze Image")

    with col2:
        st.markdown("### \U0001F4CA Analysis Report")
        output_placeholder = st.empty()
        report_area = st.empty()

    image_to_use = None
    if uploaded_file is not None:
        try:
            image_to_use = Image.open(uploaded_file).convert("RGB")
            st.image(image_to_use, caption="Uploaded image", use_column_width=True)
        except Exception as e:
            st.error(f"Could not open image: {e}")
    elif st.session_state.get("_sample_image") is not None:
        image_to_use = st.session_state.get("_sample_image")
        st.image(image_to_use, caption="Sample image", use_column_width=True)

    if analyze:
        if image_to_use is None:
            st.warning("Please upload or load a sample image before analysis.")
        else:
            try:
                with st.spinner("Loading model (this may take a while the first time)..."):
                    pipe = load_model()

                # Run analysis
                with st.container():
                    placeholder = output_placeholder
                    placeholder.markdown("**Analysis running...**\n")

                    full_response = analyze_image_with_model(pipe, image_to_use, custom_prompt)

                    # Stream the response character-by-character to mimic typing
                    typed = ""
                    for ch in full_response:
                        typed += ch
                        report_area.text(typed)
                        time.sleep(0.005)

                    placeholder.markdown("**Analysis complete**")

            except Exception as e:
                st.error(f"Error during analysis: {e}")



if __name__ == "__main__":
    main()
