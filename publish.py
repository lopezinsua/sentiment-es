"""
Publica el modelo entrenado en HuggingFace Hub.
Uso: HF_TOKEN=hf_xxx python publish.py
"""
import glob
import os
import sys

from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: Set HF_TOKEN environment variable.")
    print("  Linux/Mac: export HF_TOKEN=hf_xxx && python publish.py")
    print("  Windows:   set HF_TOKEN=hf_xxx && python publish.py")
    sys.exit(1)

login(token=token)

HF_REPO = "lopezinsua/beto-sentiment-es"
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

checkpoints = sorted(
    glob.glob("./results/checkpoint-*"),
    key=lambda p: int(p.split("-")[-1]),
)
best_ckpt = checkpoints[-1] if checkpoints else "./results"
print(f"Cargando desde: {best_ckpt}")

model = AutoModelForSequenceClassification.from_pretrained(best_ckpt)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model.push_to_hub(HF_REPO, token=token)
tokenizer.push_to_hub(HF_REPO, token=token)

print(f"\nModelo publicado en: https://huggingface.co/{HF_REPO}")
