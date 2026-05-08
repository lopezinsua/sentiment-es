"""
Publica el modelo entrenado en HuggingFace Hub.
Uso: HF_TOKEN=hf_xxx py -3.12 publish.py
"""
import os, glob, sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

token = os.environ.get("HF_TOKEN") or (sys.argv[1] if len(sys.argv) > 1 else None)
if not token:
    print("ERROR: Proporciona el token como HF_TOKEN=hf_xxx py -3.12 publish.py")
    print("   o como argumento: py -3.12 publish.py hf_xxx")
    sys.exit(1)

login(token=token)

HF_REPO = "lopezinsua/beto-sentiment-es"
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

# Carga el mejor checkpoint (el de mayor numero = ultimo guardado por load_best_model_at_end)
checkpoints = sorted(glob.glob("./results/checkpoint-*"), key=lambda p: int(p.split("-")[-1]))
best_ckpt   = checkpoints[-1] if checkpoints else "./results"
print(f"Cargando desde: {best_ckpt}")

model     = AutoModelForSequenceClassification.from_pretrained(best_ckpt)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model.push_to_hub(HF_REPO, token=token)
tokenizer.push_to_hub(HF_REPO, token=token)

print(f"\nModelo publicado en: https://huggingface.co/{HF_REPO}")
