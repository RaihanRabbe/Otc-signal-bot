import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .indicators import rsi, ema, macd, bollinger, stochastic

FEATURE_COLS = [
    'rsi_14','ema_9_rel','ema_21_rel',
    'macd_line','macd_signal','macd_hist',
    'bb_width','stoch_k','stoch_d',
    'ret_1','ret_3','ret_5','vol_norm'
]

def _clean_columns(df: pd.DataFrame):
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        raise ValueError(f'Missing required column: {names}')
    tcol = pick('time','timestamp','date','datetime')
    ocol = pick('open','o')
    ccol = pick('close','c')
    hcol = cols.get('high') or cols.get('h')
    lcol = cols.get('low') or cols.get('l')
    vcol = cols.get('volume') or cols.get('vol') or cols.get('v')

    rename_map = {tcol:'time', ocol:'open', ccol:'close'}
    if hcol: rename_map[hcol] = 'high'
    if lcol: rename_map[lcol] = 'low'
    if vcol: rename_map[vcol] = 'volume'
    out = df.rename(columns=rename_map).copy()

    if 'high' not in out.columns:
        out['high'] = out[['open','close']].max(axis=1)
    if 'low' not in out.columns:
        out['low'] = out[['open','close']].min(axis=1)
    if 'volume' not in out.columns:
        out['volume'] = 1.0

    return out[['time','open','high','low','close','volume']]

def _feature_engineer(df: pd.DataFrame):
    df = df.copy()
    df['rsi_14'] = rsi(df['close'], 14)
    df['ema_9'] = ema(df['close'], 9)
    df['ema_21'] = ema(df['close'], 21)
    df['ema_9_rel'] = (df['ema_9'] - df['close']) / (df['close'] + 1e-12)
    df['ema_21_rel'] = (df['ema_21'] - df['close']) / (df['close'] + 1e-12)

    macd_line, macd_signal, macd_hist = macd(df['close'])
    df['macd_line'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist

    ma, upper, lower, width = bollinger(df['close'])
    df['bb_width'] = width

    k, d = stochastic(df['high'], df['low'], df['close'])
    df['stoch_k'] = k
    df['stoch_d'] = d

    df['ret_1'] = df['close'].pct_change(1).fillna(0)
    df['ret_3'] = df['close'].pct_change(3).fillna(0)
    df['ret_5'] = df['close'].pct_change(5).fillna(0)

    vol_ma = df['volume'].rolling(20).mean()
    df['vol_norm'] = df['volume'] / (vol_ma + 1e-12)

    df = df.dropna().reset_index(drop=True)
    return df

def _target_next_up(df: pd.DataFrame):
    next_close = df['close'].shift(-1)
    y = (next_close > df['close']).astype(int)[:-1]
    X = df.iloc[:-1].copy()
    return X, y

def _time_series_backtest(X: pd.DataFrame, y: pd.Series, splits=5):
    n = len(X)
    fold_size = n // (splits + 1)
    metrics = []
    models = []
    for i in range(1, splits+1):
        train_end = fold_size * i
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_valid, y_valid = X.iloc[train_end: train_end+fold_size], y.iloc[train_end: train_end+fold_size]
        if len(X_valid) == 0:
            break
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train[FEATURE_COLS], y_train)
        pred = model.predict(X_valid[FEATURE_COLS])
        acc = accuracy_score(y_valid, pred)
        prec = precision_score(y_valid, pred, zero_division=0)
        rec = recall_score(y_valid, pred, zero_division=0)
        f1 = f1_score(y_valid, pred, zero_division=0)
        cm = confusion_matrix(y_valid, pred).tolist()
        metrics.append({'fold': i, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm})
        models.append(model)
    best_idx = int(np.argmax([m['accuracy'] for m in metrics]))
    return metrics, models[best_idx]

def prepare_and_backtest(raw_df: pd.DataFrame, min_rows: int = 500):
    df = _clean_columns(raw_df)
    df_feat = _feature_engineer(df)
    X, y = _target_next_up(df_feat)

    if len(X) < max(300, min_rows):
        raise ValueError('Need at least ~300 rows after feature engineering for a meaningful backtest.')

    metrics, best_model = _time_series_backtest(X, y, splits=5)

    report = {
        'rows_used': len(X),
        'fold_metrics': metrics,
        'avg_accuracy': float(np.mean([m['accuracy'] for m in metrics])),
        'trained_model': best_model
    }

    latest_feat = X.iloc[[-1]][FEATURE_COLS]
    return report, latest_feat

def infer_next_signal(model, latest_feat: pd.DataFrame, trade_threshold: float = 0.6):
    prob = model.predict_proba(latest_feat)[0][1] if hasattr(model, 'predict_proba') else None
    pred = model.predict(latest_feat)[0]
    # thresholded decision (NO TRADE mid-zone)
    decision = 'NO TRADE'
    if prob is not None:
        if prob >= trade_threshold:
            decision = 'BUY'
        elif prob <= (1.0 - trade_threshold):
            decision = 'SELL'
    else:
        decision = 'BUY' if pred == 1 else 'SELL'
    confidence = int(50 + abs((prob if prob is not None else 0.5) - 0.5) * 100) if prob is not None else 60
    return {
        'signal': decision,
        'prob_up': float(prob) if prob is not None else None,
        'confidence': confidence,
        'window': 'Next candle (1m)'
    }

