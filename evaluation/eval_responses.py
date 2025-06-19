# eval_responses.py - Evalúa respuestas generadas con métricas ROUGE y BLEU
from evaluate import load as load_metric
import json
import nltk
nltk.download('punkt')

# Ruta al archivo generado con respuestas y referencias
INPUT_PATH = "../evaluation/generated_vs_reference.jsonl"

# Leer archivo
examples = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

# Filtrar vacíos o inválidos
examples = [e for e in examples if e.get("reference") and e.get("generated")]

references = [e["reference"] for e in examples]
predictions = [e["generated"] for e in examples]

# Cargar métricas
rouge = load_metric("rouge")
bleu = load_metric("bleu")

# Calcular ROUGE
rouge_result = rouge.compute(predictions=predictions, references=references)

# Calcular BLEU (sin tokenizar manualmente)
bleu_result = bleu.compute(predictions=predictions, references=references)

# Mostrar resultados
print("===== MÉTRICAS AUTOMÁTICAS =====")
print("\nROUGE:")
for k, v in rouge_result.items():
    if hasattr(v, "fmeasure"):
        print(f"{k}: {v.fmeasure:.4f}")
    elif isinstance(v, dict) and "fmeasure" in v:
        print(f"{k}: {v['fmeasure']:.4f}")
    else:
        print(f"{k}: {v}")

print("\nBLEU:")
print(f"BLEU score: {bleu_result['bleu']:.4f}")
