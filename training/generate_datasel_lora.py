# generate_dataset_lora.py - Genera ejemplos instructivos para fine-tuning LoRA
import pandas as pd
import json
import random
chunks_es = pd.read_csv("../data/processed/chunks/chunks_es.csv")
chunks_en = pd.read_csv("../data/processed/chunks/chunks_eng.csv")



# Generadores de preguntas
prompts_es = [
    "¿Qué derechos tengo según este texto?",
    "¿Qué debo hacer si esto me sucede?",
    "¿Cuál es la recomendación principal?",
    "¿Qué dice este texto sobre ICE?",
    "¿Cómo debo reaccionar ante esta situación?"
]
prompts_en = [
    "What rights do I have according to this text?",
    "What should I do if this happens to me?",
    "What is the main recommendation?",
    "What does this say about ICE?",
    "How should I respond to this situation?"
]

# Mezclar ambos
all_chunks = pd.concat([chunks_es, chunks_en], ignore_index=True)

examples = []
for _, row in all_chunks.iterrows():
    text = row['text'].strip()
    is_es = row['chunk_id'].startswith("es")
    question = random.choice(prompts_es if is_es else prompts_en)
    examples.append({
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": text}
        ]
    })

# Guardar dataset
with open("../data/processed/finetune/llama_finetune_500examples.jsonl", "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print("✅ Dataset generado: llama_finetune_500examples.jsonl")
