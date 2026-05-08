# beto-sentiment-es

Fine-tuning de `dccuchile/bert-base-spanish-wwm-cased` (BETO) para clasificacion de sentimiento en tres clases: positivo, negativo y neutro. Entrenado sobre tweets reales en español con anotacion humana.

## Resultados

| Metrica  | Valor      |
|----------|------------|
| Accuracy | **67.24%** |
| F1 macro | **67.12%** |

Classification report (test set, 870 ejemplos):

| Clase    | Precision | Recall | F1   |
|----------|-----------|--------|------|
| negative | 0.73      | 0.73   | 0.73 |
| neutral  | 0.58      | 0.55   | 0.57 |
| positive | 0.71      | 0.74   | 0.72 |

## Uso

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="lopezinsua/beto-sentiment-es")
classifier("Me encanta este producto, es increible")
# [{'label': 'positive', 'score': 0.992}]
```

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model     = AutoModelForSequenceClassification.from_pretrained("lopezinsua/beto-sentiment-es")
tokenizer = AutoTokenizer.from_pretrained("lopezinsua/beto-sentiment-es")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label = model.config.id2label[probs.argmax().item()]
    return {"label": label, "confidence": round(probs.max().item(), 3)}
```

## Dataset y entrenamiento

- **Dataset**: `mteb/tweet_sentiment_multilingual` (español) — 1839 tweets, clases balanceadas (613 por clase)
- **Modelo base**: `dccuchile/bert-base-spanish-wwm-cased` (BETO)
- **Epochs**: 5 — mejor checkpoint en epoch 4
- **Batch size**: 16 | **Learning rate**: 2e-5

## Requisitos

```
pip install -r requirements.txt
```

## Limitaciones y posibles mejoras

El cuello de botella es el dataset: 1839 ejemplos son suficientes para el pipeline pero insuficientes para produccion. La clase neutral (F1 0.57) es la mas debil porque las frases ambiguas requieren mas datos para aprender el limite de decision.

Mejoras que no implemente porque el objetivo era el pipeline:

- **Mas datos**: con ~10.000 ejemplos anotados BETO llegaria a 80-82%. Fuentes: TASS, resenas de Amazon ES, tweets propios.
- **Data augmentation**: traduccion automatica del dataset ingles al español para multiplicar ejemplos sin coste de anotacion.
- **Modelo mas grande**: `PlanTL-GOB-ES/roberta-base-bne` con el mismo dataset daria ~2-4 puntos mas.
- **Clasificacion binaria**: eliminar neutral sube el accuracy a ~85-90%, que es lo util en la mayoria de casos de negocio.

## Autor

López Insua — [github.com/lopezinsua](https://github.com/lopezinsua) | [huggingface.co/lopezinsua](https://huggingface.co/lopezinsua)
