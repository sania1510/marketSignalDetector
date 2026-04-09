# models/nlp_sentiment.py
import os
import torch
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding='utf-8')
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

MODEL_NAME  = "ProsusAI/finbert"   # pre-trained FinBERT on HuggingFace
MODEL_SAVE  = "models/saved/finbert_finetuned"
LABEL_MAP   = {0: "negative", 1: "neutral", 2: "positive"}
SCORE_MAP   = {0: -1.0,       1: 0.0,       2: 1.0}  # for numeric signal use


class FinancialDataset(Dataset):
    """PyTorch dataset wrapping tokenized PhraseBank sentences."""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


class SentimentModel:
    """
    Wraps FinBERT for fine-tuning and inference.
    Usage:
        model = SentimentModel()
        model.fine_tune(phrasebank_df)    # one-time training
        scores = model.score_headlines(headlines_df)
    """

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SentimentModel] Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fine_tune(self, df: pd.DataFrame,
                  epochs: int = 3,
                  batch_size: int = 16,
                  lr: float = 2e-5):
        """
        Fine-tunes FinBERT on the PhraseBank dataset.
        df must have columns: 'text', 'label' (0=neg, 1=neutral, 2=pos)
        """
        print(f"[SentimentModel] Fine-tuning on {len(df)} examples...")

        X_train, X_val, y_train, y_val = train_test_split(
            df["text"], df["label"],
            test_size=0.15, stratify=df["label"], random_state=42
        )

        train_ds = FinancialDataset(X_train, y_train, self.tokenizer)
        val_ds   = FinancialDataset(X_val,   y_val,   self.tokenizer)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        # Load base FinBERT with a 3-class classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=3
        ).to(self.device)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )

        for epoch in range(epochs):
            # --- Training pass ---
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids      = batch["input_ids"].to(self.device),
                    attention_mask = batch["attention_mask"].to(self.device),
                    labels         = batch["labels"].to(self.device)
                )
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += outputs.loss.item()

            avg_loss = total_loss / len(train_loader)

            # --- Validation pass ---
            val_acc = self._evaluate(val_loader)
            print(f"  Epoch {epoch+1} | loss: {avg_loss:.4f} | val_acc: {val_acc:.3f}")

        # Save for reuse
        self._save()
        print("[SentimentModel] Fine-tuning complete.")

    def _evaluate(self, loader) -> float:
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                logits = self.model(
                    input_ids      = batch["input_ids"].to(self.device),
                    attention_mask = batch["attention_mask"].to(self.device)
                ).logits
                preds = logits.argmax(dim=1).cpu()
                correct += (preds == batch["labels"]).sum().item()
                total   += len(batch["labels"])
        return correct / total

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def load(self):
        """Loads a previously saved fine-tuned model."""
        print(f"[SentimentModel] Loading saved model from {MODEL_SAVE}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_SAVE
        ).to(self.device)
        self.model.eval()

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Returns a list of dicts: {label, score, confidence}
        score: -1 (negative), 0 (neutral), +1 (positive)
        """
        if self.model is None:
            self.load()

        self.model.eval()
        results = []
        # Process in small batches to avoid OOM on CPU
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch_texts,
                truncation=True, padding=True,
                max_length=128, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits
                probs  = torch.softmax(logits, dim=1).cpu().numpy()

            for prob in probs:
                pred_idx = prob.argmax()
                results.append({
                    "label":      LABEL_MAP[pred_idx],
                    "score":      SCORE_MAP[pred_idx],
                    "confidence": float(prob[pred_idx]),
                    "prob_neg":   float(prob[0]),
                    "prob_neu":   float(prob[1]),
                    "prob_pos":   float(prob[2]),
                })
        return results

    def score_headlines(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes the headlines DataFrame, runs inference, aggregates to daily scores.
        Returns a DataFrame indexed by date with columns:
            sentiment_mean, sentiment_bullish_ratio, headline_count
        """
        print(f"[SentimentModel] Scoring {len(news_df)} headlines...")

        texts = (news_df["title"].fillna("") + " " +
                 news_df["description"].fillna("")).tolist()

        predictions = self.predict(texts)
        news_df = news_df.copy()
        news_df["sentiment_score"]  = [p["score"]      for p in predictions]
        news_df["confidence"]       = [p["confidence"] for p in predictions]
        news_df["prob_pos"]         = [p["prob_pos"]   for p in predictions]
        news_df["prob_neg"]         = [p["prob_neg"]   for p in predictions]

        # Aggregate to daily level
        daily = news_df.groupby("date").agg(
            sentiment_mean      = ("sentiment_score", "mean"),
            sentiment_std       = ("sentiment_score", "std"),
            bullish_ratio       = ("prob_pos", "mean"),   # avg P(positive)
            bearish_ratio       = ("prob_neg", "mean"),   # avg P(negative)
            headline_count      = ("title", "count"),
            avg_confidence      = ("confidence", "mean"),
        ).reset_index()
        daily["sentiment_strength"] = daily["bullish_ratio"] - daily["bearish_ratio"]

        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.set_index("date")

        # Smoothed 3-day rolling sentiment (reduces noise)
        daily["sentiment_3d_ma"] = (daily["sentiment_mean"]
                                    .rolling(3, min_periods=1).mean())

        out_path = "data/processed/sentiment_daily.csv"
        daily.to_csv(out_path)
        print(f"[SentimentModel] Saved daily sentiment → {out_path}")
        return daily

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        os.makedirs(MODEL_SAVE, exist_ok=True)
        self.model.save_pretrained(MODEL_SAVE)
        self.tokenizer.save_pretrained(MODEL_SAVE)
        print(f"[SentimentModel] Model saved → {MODEL_SAVE}")