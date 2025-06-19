from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Ruta a tu dataset jsonl
from pathlib import Path

# Encuentra la raíz del proyecto (ajusta si cambias estructura)
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "finetune" / "llama_finetune_4000examples_clean_v2.jsonl"

MODEL_NAME = "microsoft/phi-2"
raw_dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")

# Tokenizador
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Preprocesamiento
def preprocess(example):
    turns = example["conversations"]
    conversation = ""
    for turn in turns:
        role = turn["role"]
        content = turn["content"].strip()
        if role == "user":
            conversation += f"<|user|> {content}\n"
        elif role == "assistant":
            conversation += f"<|assistant|> {content}\n"
    tokenized = tokenizer(conversation, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Aplicar transformación
processed_dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)

# Modelo base y LoRA
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,  # reducido para evitar sobreajuste
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.15,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Collator y argumentos de entrenamiento
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./models/phi2_defense_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=10,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    logging_dir="./logs_defense",
    report_to="none",
    remove_unused_columns=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# Guardar adaptadores LoRA
model.save_pretrained("./models/phi2_defense_lora_adapters")
print("✅ Fine-tuning completo con dataset limpio.")