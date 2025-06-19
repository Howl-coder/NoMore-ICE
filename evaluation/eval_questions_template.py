# eval_questions_template.py - Genera preguntas y referencias desde chunks
import pandas as pd
import json
import random

chunks_es = pd.read_csv("../data/processed/chunks/chunks_es.csv")
chunks_en = pd.read_csv("../data/processed/chunks/chunks_eng.csv")

prompts_es = [
    "¿Qué derechos tengo según este texto?",
    "¿Qué debo hacer si esto me sucede?",
    "¿Cuál es la recomendación principal?"
]
prompts_en = [
    "What rights do I have according to this text?",
    "What should I do if this happens to me?",
    "What is the main recommendation?"
]

all_chunks = pd.concat([chunks_es, chunks_en], ignore_index=True).sample(100, random_state=42)

pairs = []
for _, row in all_chunks.iterrows():
    text = row['text'].strip()
    is_es = row['chunk_id'].startswith("es")
    question = random.choice(prompts_es if is_es else prompts_en)
    pairs.append({
        "question": question,
        "reference": text
    })
with open("../evaluation/eval_questions.jsonl", "w", encoding="utf-8") as f:

    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print("✅ Archivo de evaluación generado: eval_questions.jsonl")
