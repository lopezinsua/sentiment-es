"""
sentiment-es: Fine-tuning BETO para clasificacion de sentimiento en espanol.

Dataset: mteb/tweet_sentiment_multilingual (tweets reales en espanol, anotacion humana).
Modelo base: dccuchile/bert-base-spanish-wwm-cased (BETO).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
\n
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
RESULTS_DIR = Path(__file__).parent / "results"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def predict(text: str, model, tokenizer, device) -> dict:
    """Run inference on a single text. Model must be in eval mode."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label = model.config.id2label[probs.argmax().item()]
    return {
        "label": label,
        "confidence": round(probs.max().item(), 3),
        "scores": {model.config.id2label[i]: round(p.item(), 3) for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    # ── 1. GPU check ──────────────────────────────────────────────────────────
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    print("\nCargando dataset...")
    raw = load_dataset("mteb/tweet_sentiment_multilingual", "spanish")
    print(raw)

    def cast_label(batch):
        batch["label"] = [int(lbl) for lbl in batch["label"]]
        return batch

    raw = raw.map(cast_label, batched=True)
    train_data = raw["train"]
    test_data = raw["test"]

    print(f"\nTrain: {len(train_data)} | Test: {len(test_data)}")
    counts = Counter(train_data["label"])
    print("Distribucion de clases (train):")
    for lid in sorted(counts):
        cnt = counts[lid]
        print(f"  {ID2LABEL[lid]}: {cnt} ({cnt/len(train_data)*100:.1f}%)")

    # ── 3. Tokenizacion ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")

    keep_cols = {"label"}
    train_tok = train_data.map(
        tokenize, batched=True,
        remove_columns=[c for c in train_data.column_names if c not in keep_cols],
    )
    test_tok = test_data.map(
        tokenize, batched=True,
        remove_columns=[c for c in test_data.column_names if c not in keep_cols],
    )
    train_tok.set_format("torch")
    test_tok.set_format("torch")

    # ── 4. Fine-tuning ────────────────────────────────────────────────────────
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        compute_metrics=compute_metrics,
    )

    print("\n=== Entrenando ===")
    train_result = trainer.train()
    print(f"Loss final: {train_result.training_loss:.4f}")

    # ── 5. Evaluacion ─────────────────────────────────────────────────────────
    eval_results = trainer.evaluate()
    accuracy = eval_results["eval_accuracy"]
    f1 = eval_results["eval_f1"]
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 macro: {f1:.4f}")

    preds_out = trainer.predict(test_tok)
    y_pred = np.argmax(preds_out.predictions, axis=-1)
    y_true = preds_out.label_ids

    report = classification_report(
        y_true, y_pred, target_names=[ID2LABEL[i] for i in sorted(ID2LABEL)]
    )
    print("\n=== Classification Report ===")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=[ID2LABEL[i] for i in sorted(ID2LABEL)])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — beto-sentiment-es")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Guardado: confusion_matrix.png")

    history = trainer.state.log_history
    train_losses = [
        (h["step"], h["loss"]) for h in history if "loss" in h and "eval_loss" not in h
    ]
    eval_losses = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]

    fig, ax = plt.subplots(figsize=(8, 4))
    if train_losses:
        steps, losses = zip(*train_losses)
        ax.plot(steps, losses, label="train loss", alpha=0.7)
    if eval_losses:
        steps, elosses = zip(*eval_losses)
        ax.plot(steps, elosses, "o-", label="eval loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Eval Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("Guardado: loss_curve.png")

    metrics = {"accuracy": round(accuracy, 4), "f1": round(f1, 4)}
    with open("metrics.json", "w") as fh:
        json.dump(metrics, fh)
    print(f"\nMetricas guardadas: {metrics}")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ejemplos = [
        "Me encanta este producto, es increible",
        "El servicio fue pesimo, nunca volveria",
        "El producto llego en el plazo indicado",
        "Llevo esperando tres semanas y nada, vergonzoso",
        "Justo lo que necesitaba, muy recomendable",
        "No esta mal, cumple con lo basico",
    ]

    print("\n=== Ejemplos finales ===")
    print(f"{'TEXTO':<50} {'LABEL':<12} {'CONF':>6}")
    print("-" * 72)
    for texto in ejemplos:
        res = predict(texto, model, tokenizer, device)
        print(f"{texto[:48]:<50} {res['label']:<12} {res['confidence']:>6.3f}")

    print("\nEntrenamiento completo. Ejecuta publish.py para subir a HuggingFace.")
