import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import torch
from tqdm import tqdm

# Rutas del modelo
BASE_MODEL = "microsoft/phi-2"
LORA_PATH = Path(__file__).resolve().parent.parent / "training" / "models" / "phi2_lora_adapters"
QUESTIONS_FILE = "../evaluation/eval_questions.jsonl"
OUTPUT_FILE = "../evaluation/generated_vs_reference.jsonl"

# Cargar modelo y tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
model.eval()

# Leer preguntas de entrada
examples = []
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

# Generar respuestas con barra de progreso
generated = []
for i, ex in enumerate(tqdm(examples, desc="Generando respuestas")):
    prompt = f"<|user|> {ex['question']}\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text = decoded.split("<|assistant|>")[-1].strip()

    generated.append({
        "question": ex["question"],
        "reference": ex["reference"],
        "generated": generated_text
    })

# Guardar archivo
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for row in generated:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"✅ Archivo generado para evaluación: {OUTPUT_FILE}")
