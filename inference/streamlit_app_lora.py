# streamlit_app_lora.py - Chat local con Phi-2 LoRA

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from pathlib import Path
st.set_page_config(page_title="DefensaAI LoRA", layout="wide")
# Configuración general
BASE_MODEL = "microsoft/phi-2"
LORA_PATH = Path(__file__).resolve().parent.parent / "training" / "models" / "phi2_lora_adapters"

def load_model():
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Cargar modelo base
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Cargar adaptadores LoRA
    model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
    model.eval()
    return tokenizer, model


# Cargar modelo y tokenizer
st.session_state.setdefault("tokenizer", None)
st.session_state.setdefault("model", None)
if st.session_state["model"] is None:
    with st.spinner("Cargando modelo ajustado..."):
        tokenizer, model = load_model()
        st.session_state["tokenizer"] = tokenizer
        st.session_state["model"] = model

# Interfaz de usuario

st.title("🧠 DefensaAI - LLM entrenado localmente")
st.markdown("Responde preguntas legales con tu modelo Phi-2 ajustado con LoRA.")

user_input = st.text_area("✍️ Escribe tu pregunta aquí:", height=120)
if st.button("Responder") and user_input.strip():
    tokenizer = st.session_state["tokenizer"]
    model = st.session_state["model"]

    # Construir prompt estilo entrenamiento
    prompt = f"<|user|> {user_input}\n<|assistant|>"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.split("<|assistant|>")[-1].strip()

    st.success("✅ Respuesta del modelo ajustado:")
    st.markdown(response)
