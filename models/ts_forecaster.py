# models/ts_forecaster.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

MODEL_SAVE  = "models/saved/lstm_forecaster.pt"
SCALER_SAVE = "models/saved/feature_scaler.pkl"
SEQUENCE_LEN = 20   # look-back window

FEATURE_COLS = [
    "return_1d", "return_5d", "return_20d",
    "rsi_normalized", "macd_norm",
    "bb_position", "bb_width",
    "volume_ratio", "vol_price_corr",
    "volatility_10d", "volatility_30d",
    "high_vol_regime",
    "price_vs_sma10", "price_vs_sma50", "ma_cross",
    "VIX", "YieldCurve",
    "sentiment_3d_ma",
    "bullish_ratio",
]


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out)
        return self.fc(out[:, -1, :]).squeeze(1)


class ForecastModel:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None
        print(f"[ForecastModel] Device: {self.device}")

    # ---------------- DATA ---------------- #

    def prepare_data(self, master_df, sentiment_df):
        from features.technical import TechnicalFeatures

        df = master_df[["SPY"]].rename(columns={"SPY": "Close"}).copy()

        spy_raw = pd.read_csv("data/raw/prices/SPY.csv",
                              index_col=0, parse_dates=True)
        df = df.join(spy_raw[["Open", "High", "Low", "Volume"]], how="left")

        for col in ["VIX", "YieldCurve"]:
            if col in master_df.columns:
                df[col] = master_df[col]

        tf = TechnicalFeatures()
        df = tf.add_all(df)

        # Merge sentiment
        df = df.join(sentiment_df[["sentiment_3d_ma", "bullish_ratio"]], how="left")
        df["sentiment_3d_ma"] = df["sentiment_3d_ma"].fillna(0)
        df["bullish_ratio"]   = df["bullish_ratio"].fillna(0.33)

        #  IMPORTANT FIX
        df = df.dropna(subset=["target_3d"])
        df = df.dropna()

        print(f"[ForecastModel] Clean dataset: {len(df)} rows, {df.shape[1]} cols")
        return df

    def _make_sequences(self, df):
        if "target_3d" not in df.columns:
            raise ValueError("target_3d missing — check technical.py")

        available = [c for c in FEATURE_COLS if c in df.columns]

        X_raw = df[available].values
        y_raw = df["target_3d"].values  # 

        X_scaled = self.scaler.transform(X_raw)

        X, y = [], []
        for i in range(SEQUENCE_LEN, len(X_scaled)):
            X.append(X_scaled[i - SEQUENCE_LEN:i])
            y.append(y_raw[i])

        return np.array(X), np.array(y)

    # ---------------- TRAIN ---------------- #

    def train(self, df, epochs=40, batch_size=32, lr=1e-3, val_split=0.15):
        available = [c for c in FEATURE_COLS if c in df.columns]

        X_raw = df[available].values
        self.scaler.fit(X_raw)

        split_idx = int(len(df) * (1 - val_split))
        train_df = df.iloc[:split_idx]
        val_df   = df.iloc[split_idx - SEQUENCE_LEN:]

        X_tr, y_tr = self._make_sequences(train_df)
        X_va, y_va = self._make_sequences(val_df)

        print(f"[ForecastModel] Train: {len(X_tr)} | Val: {len(X_va)} sequences")

        train_loader = DataLoader(SequenceDataset(X_tr, y_tr),
                                  batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(SequenceDataset(X_va, y_va),
                                  batch_size=batch_size)

        self.model = LSTMForecaster(input_size=X_tr.shape[2]).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion  = nn.BCELoss()

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = self.model(xb.to(self.device))
                loss = criterion(pred, yb.to(self.device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss, preds, labels = 0, [], []

            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = self.model(xb.to(self.device))
                    val_loss += criterion(pred, yb.to(self.device)).item()
                    preds.extend((pred.cpu() > 0.5).int().tolist())
                    labels.extend(yb.int().tolist())

            val_acc = accuracy_score(labels, preds)
            avg_vl = val_loss / len(val_loader)

            if avg_vl < best_val_loss:
                best_val_loss = avg_vl
                self._save()
                tag = " ← best"
            else:
                tag = ""

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | loss: {train_loss/len(train_loader):.4f} | val_acc: {val_acc:.3f}{tag}")

        print("[ForecastModel] Training complete.")
        self.load()

    # ---------------- INFERENCE ---------------- #

    def predict_proba(self, sequence):
        self.model.eval()
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.model(x).cpu())

    def predict_latest(self, df):
        available = [c for c in FEATURE_COLS if c in df.columns]
        X_scaled = self.scaler.transform(df[available].values)

        seq = X_scaled[-SEQUENCE_LEN:]
        prob_up = self.predict_proba(seq)

        return {
            "prob_up": prob_up,
            "direction": "UP" if prob_up > 0.5 else "DOWN",
            "confidence": abs(prob_up - 0.5) * 2
        }

    # ---------------- SAVE / LOAD ---------------- #

    def _save(self):
        os.makedirs("models/saved", exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_SAVE)
        joblib.dump(self.scaler, SCALER_SAVE)

    def load(self):
        self.scaler = joblib.load(SCALER_SAVE)
        n_features = self.scaler.n_features_in_
        self.model = LSTMForecaster(n_features).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_SAVE, map_location=self.device))
        self.model.eval()
        print("[ForecastModel] Model loaded.")