# infer_single.py - Prueba rápida desde consola con modelo LoRA
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import torch

BASE_MODEL = "microsoft/phi-2"
LORA_PATH = Path(__file__).resolve().parent.parent / "training" / "models" / "phi2_lora_adapters"

# Cargar modelo
print("Cargando modelo...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
model.eval()

while True:
    query = input("🧠 Pregunta (enter para salir): ")
    if not query.strip():
        break

    prompt = f"<|user|> {query.strip()}\n<|assistant|>"
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
    print("\n📝 Respuesta:")
    print(decoded.split("<|assistant|>")[-1].strip())
    print("="*60)
