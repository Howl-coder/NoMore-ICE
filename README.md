🧠 NoMoreICE — Defense AI for Immigrants
A bilingual (English-Spanish) legal assistant powered by a fine-tuned LLM to help Mexican immigrants understand and assert their rights during ICE-related encounters.

What is this project?

**NoMoreICE** is a local AI assistant trained with a curated bilingual dataset extracted from legal documents, flyers, and advocacy materials. It uses a lightweight version of the `phi-2` model with **LoRA fine-tuning** to answer questions such as:

- “Do I have to let ICE into my home?”
- “Can I film ICE officers?”
- “What should I do if ICE knocks on my door?”

- It aims to **empower immigrants** by giving them quick, safe access to know-your-rights information even offline.

---

### 🛠️ Tech stack

- `phi-2` base model (Microsoft)
- LoRA adapters (PEFT library)
- Transformers (Hugging Face)
- Streamlit for local chat UI
- Fine-tuning on legal chunks (4k examples)
- Evaluation: ROUGE, BLEU

---

### ▶️ How to run it locally

```bash
# 1. Clone the repo
git clone https://github.com/your_username/NoMoreICE.git
cd NoMoreICE

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # or source .venv/bin/activate on Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run local chat
streamlit run inference/streamlit_app_lora.py
