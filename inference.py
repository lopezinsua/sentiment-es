"""
Standalone inference from HuggingFace Hub. No training required.
Usage: python inference.py "Texto a clasificar"
"""
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "lopezinsua/beto-sentiment-es"
_model = _tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        _model.eval()
        _model.to(_device)


def predict(text: str) -> dict:
    _load()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label = _model.config.id2label[probs.argmax().item()]
    return {
        "label": label,
        "confidence": round(probs.max().item(), 3),
        "scores": {_model.config.id2label[i]: round(p.item(), 3) for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    texts = sys.argv[1:] or ["Me encanta este producto", "El servicio fue pesimo"]
    for t in texts:
        r = predict(t)
        print(f"{t[:50]:<52} {r['label']:<12} {r['confidence']:.3f}")
