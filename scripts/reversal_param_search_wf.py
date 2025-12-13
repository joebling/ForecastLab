"""
改进版参数搜索 + walk-forward 预测评估脚本（精简版）
- 仅做分类评估：预测 vs 真实标签，对每个 OOS 折计算 precision/recall/f1（宏平均）
- 不再进行交易模拟、净值/Sharpe/回撤计算，也不输出 equity_cur
- 输出 summary 仅保留 candidate、T、X、folds、f1_median、precision_median、recall_median
- 另存每候选的每折评估明细 fold_metrics.csv，含每折指标与类分布
"""

import argparse
import os
import json
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from datetime import datetime

try:
    import lightgbm as lgb
    LGB_INSTALLED = True
except Exception:
    LGB_INSTALLED = False

# ------------------------ utilities ------------------------

def ensure_dirs(out_root: str):
    os.makedirs(out_root, exist_ok=True)


def _detect_data_source(df: pd.DataFrame, data_path: str) -> str:
    # best-effort
    p = (data_path or '').lower()
    if 'binance' in p:
        return 'binance'
    if 'yahoo' in p:
        return 'yahoo'
    # infer by columns
    if 'open_time_ms' in df.columns or 'taker_buy_base_asset_volume' in df.columns:
        return 'binance'
    return 'unknown'


def _write_meta(out_root: str, meta: Dict):
    meta_path = os.path.join(out_root, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _safe_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - mu) / (sd + 1e-12)


def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['ret1'] = df['close'].pct_change()
    df['r1'] = df['ret1']
    df['ma7'] = df['close'].rolling(7).mean()
    df['ma21'] = df['close'].rolling(21).mean()
    df['vol21'] = df['ret1'].rolling(21).std()
    # RSI 14
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(com=13).mean()
    roll_down = down.ewm(com=13).mean()
    rs = roll_up / (roll_down + 1e-8)
    df['rsi_14'] = 100 - 100 / (1 + rs)
    # simple macd histogram
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd_hist'] = (ema12 - ema26) - ((ema12 - ema26).ewm(span=9).mean())
    df = df.dropna().reset_index(drop=True)
    return df


def compute_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binance flow/microstructure features derived from trading activity fields."""
    df = df.copy()

    required = [
        'volume',
        'quote_asset_volume',
        'number_of_trades',
        'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"flow feature_set requires Binance columns missing: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # buy ratio (aggressive buy share)
    df['taker_buy_base_ratio'] = df['taker_buy_base_asset_volume'] / (df['volume'] + 1e-12)
    df['taker_buy_quote_ratio'] = df['taker_buy_quote_asset_volume'] / (df['quote_asset_volume'] + 1e-12)

    # trade intensity
    df['trades_per_volume'] = df['number_of_trades'] / (df['volume'] + 1e-12)
    df['quote_per_trade'] = df['quote_asset_volume'] / (df['number_of_trades'] + 1e-12)

    # changes
    for c in [
        'taker_buy_base_ratio',
        'taker_buy_quote_ratio',
        'trades_per_volume',
        'quote_per_trade',
        'number_of_trades',
        'quote_asset_volume',
    ]:
        df[f'{c}_chg1'] = df[c].pct_change()

    # rolling z-scores (short/medium)
    for w in [7, 21]:
        df[f'taker_buy_base_ratio_z{w}'] = _safe_zscore(df['taker_buy_base_ratio'], w)
        df[f'number_of_trades_z{w}'] = _safe_zscore(df['number_of_trades'], w)
        df[f'quote_asset_volume_z{w}'] = _safe_zscore(df['quote_asset_volume'], w)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


def _select_features(df: pd.DataFrame, feature_set: str) -> List[str]:
    baseline = ['rsi_14', 'ma7', 'ma21', 'vol21', 'macd_hist', 'ret1']
    if feature_set == 'baseline':
        return baseline
    if feature_set == 'flow':
        # baseline + flow
        flow = [
            'taker_buy_base_ratio',
            'taker_buy_quote_ratio',
            'trades_per_volume',
            'quote_per_trade',
            'taker_buy_base_ratio_chg1',
            'taker_buy_quote_ratio_chg1',
            'number_of_trades_chg1',
            'quote_asset_volume_chg1',
            'taker_buy_base_ratio_z7',
            'taker_buy_base_ratio_z21',
            'number_of_trades_z7',
            'number_of_trades_z21',
            'quote_asset_volume_z7',
            'quote_asset_volume_z21',
        ]
        feats = baseline + flow
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise ValueError(f"selected features missing from dataframe: {missing}")
        return feats
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _parse_candidates_file(path: str) -> List[str]:
    if not path:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"candidates_file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines()]
    # allow empty lines and comments
    return [ln for ln in lines if ln and not ln.startswith('#')]


def _top_candidates_from_summary(summary_csv: str, top_k: int) -> List[str]:
    if not summary_csv or not os.path.exists(summary_csv):
        raise FileNotFoundError(f"summary_csv not found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    if df.empty:
        return []
    if 'candidate' not in df.columns:
        raise ValueError("summary_csv must have 'candidate' column")
    sort_col = 'f1_median' if 'f1_median' in df.columns else df.columns[-1]
    df = df.sort_values(sort_col, ascending=False)
    return df['candidate'].head(int(top_k)).tolist()


def compute_labels(df: pd.DataFrame, T: int, X: float) -> pd.DataFrame:
    df = df.copy()
    T = int(T)
    future_max = df['close'].shift(-1).rolling(T).max()
    future_min = df['close'].shift(-1).rolling(T).min()
    future_drawdown = (df['close'] - future_min) / df['close']
    future_upswing = (future_max - df['close']) / df['close']
    df['future_drawdown'] = future_drawdown
    df['future_upswing'] = future_upswing
    df['label_top'] = (future_drawdown >= X).astype(int)
    df['label_bottom'] = (future_upswing >= X).astype(int)
    df['label'] = 0
    df.loc[df['label_top'] == 1, 'label'] = -1
    df.loc[df['label_bottom'] == 1, 'label'] = 1
    return df


def walk_forward_splits(n_samples: int, init_train: int, oos_window: int, step: int) -> List[Tuple[slice, slice]]:
    splits = []
    start = init_train
    while start + oos_window <= n_samples:
        train_slice = slice(0, start)
        oos_slice = slice(start, start + oos_window)
        splits.append((train_slice, oos_slice))
        start += step
    return splits


def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: Dict = None):
    if LGB_INSTALLED:
        lgb_train = lgb.Dataset(X_train, label=y_train)
        params = params or {'objective': 'multiclass', 'num_class': 3, 'verbosity': -1}
        model = lgb.train(params, lgb_train, num_boost_round=100)
        return ('lgb', model)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train.astype(int))
        return ('rf', clf)


def predict_model(model_tuple, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    kind, model = model_tuple
    if kind == 'lgb':
        probs = model.predict(X)
        preds = np.argmax(probs, axis=1) - 1
        return preds, probs
    else:
        probs_raw = model.predict_proba(X)
        classes = list(model.classes_)
        probs = np.zeros((X.shape[0], 3))
        for cls in [0, 1, 2]:
            if cls in classes:
                probs[:, cls] = probs_raw[:, classes.index(cls)]
            else:
                probs[:, cls] = 0.0
        preds = model.predict(X).astype(int) - 1
        return preds, probs


# ------------------------ main flow ------------------------

def main(args):
    ensure_dirs(args.out_root)

    # read once for columns
    head_cols = pd.read_csv(args.data_path, nrows=1).columns
    parse_col = 'Date' if 'Date' in head_cols else 'timestamp'
    df = pd.read_csv(args.data_path, parse_dates=[parse_col])

    # normalize column names
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'timestamp', 'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)
    if 'timestamp' not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    data_source = _detect_data_source(df, args.data_path)

    # features
    df_features = compute_basic_features(df)
    if args.feature_set == 'flow':
        # attach flow features computed from Binance activity fields
        # compute_flow_features expects those columns to exist
        df_features = compute_flow_features(df_features)

    features = _select_features(df_features, args.feature_set)

    # candidates control (for stage-2)
    stage = int(args.stage)
    candidates_filter: List[str] = []
    if stage == 2:
        if args.candidates_file:
            candidates_filter = _parse_candidates_file(args.candidates_file)
        elif args.baseline_summary and args.top_k > 0:
            candidates_filter = _top_candidates_from_summary(args.baseline_summary, args.top_k)
        else:
            raise ValueError("stage=2 requires --candidates_file OR (--baseline_summary and --top_k)")

    # write run meta for webapp
    meta = {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_path': args.data_path,
        'data_source': data_source,
        'feature_set': args.feature_set,
        'stage': stage,
        'T_list': list(map(int, args.T_list)),
        'X_list': list(map(float, args.X_list)),
        'init_train': int(args.init_train),
        'oos_window': int(args.oos_window),
        'step': int(args.step),
        'model_preference': 'lightgbm' if LGB_INSTALLED else 'random_forest',
        'lgb_installed': bool(LGB_INSTALLED),
        'selected_features': features,
    }
    if stage == 2:
        meta.update({
            'baseline_summary': args.baseline_summary,
            'top_k': int(args.top_k),
            'candidates_file': args.candidates_file,
            'candidates_used': candidates_filter,
        })
    _write_meta(args.out_root, meta)

    results = []

    for T in args.T_list:
        for X in args.X_list:
            cand_name = f"T{T}_X{int(X*100)}"
            if candidates_filter and cand_name not in candidates_filter:
                continue

            print(f"Processing candidate: {cand_name}")
            df_lab = compute_labels(df_features, T, X).dropna().reset_index(drop=True)

            X_all = df_lab[features]
            y_all = df_lab['label']
            n = len(df_lab)
            splits = walk_forward_splits(n, args.init_train, args.oos_window, args.step)
            cand_metrics = []

            for fold_idx, (train_slice, oos_slice) in enumerate(splits):
                X_train, y_train = X_all.iloc[train_slice], y_all.iloc[train_slice]
                X_oos, y_oos = X_all.iloc[oos_slice], y_all.iloc[oos_slice]

                # align labels to 0,1,2 for classifier
                map_train = y_train + 1

                model = train_model(X_train, map_train)
                preds_fold, _ = predict_model(model, X_oos)
                preds_fold = np.array(preds_fold).astype(int)

                # classification metrics (on oos)
                try:
                    labels_all = [-1, 0, 1]
                    prec = precision_score(y_oos, preds_fold, average='macro', labels=labels_all, zero_division=0)
                    rec = recall_score(y_oos, preds_fold, average='macro', labels=labels_all, zero_division=0)
                    f1 = f1_score(y_oos, preds_fold, average='macro', labels=labels_all, zero_division=0)
                    acc = accuracy_score(y_oos, preds_fold)
                    bacc = rec
                except Exception:
                    prec = rec = f1 = acc = bacc = 0

                vc_true = y_oos.value_counts().to_dict()
                vc_pred = pd.Series(preds_fold).value_counts().to_dict()

                def g(d, k):
                    return int(d.get(k, 0))

                cand_metrics.append({
                    'fold': fold_idx, 'T': T, 'X': X,
                    'precision': prec, 'recall': rec, 'f1': f1,
                    'accuracy': acc, 'balanced_accuracy': bacc,
                    'y_true_-1': g(vc_true, -1), 'y_true_0': g(vc_true, 0), 'y_true_1': g(vc_true, 1),
                    'y_pred_-1': g(vc_pred, -1), 'y_pred_0': g(vc_pred, 0), 'y_pred_1': g(vc_pred, 1)
                })

            df_cand = pd.DataFrame(cand_metrics)
            agg = df_cand.agg({'precision': ['median'], 'recall': ['median'], 'f1': ['mean', 'median'], 'accuracy': ['median'], 'balanced_accuracy': ['median']})
            result = {
                'candidate': cand_name, 'T': T, 'X': X,
                'folds': int(len(cand_metrics)),
                'precision_median': float(agg['precision']['median']),
                'recall_median': float(agg['recall']['median']),
                'f1_median': float(agg['f1']['median']),
                'accuracy_median': float(agg['accuracy']['median']),
                'balanced_accuracy_median': float(agg['balanced_accuracy']['median'])
            }
            results.append(result)

            cand_dir = os.path.join(args.out_root, cand_name)
            os.makedirs(cand_dir, exist_ok=True)
            df_cand.to_csv(os.path.join(cand_dir, 'fold_metrics.csv'), index=False)

    res_df = pd.DataFrame(results).sort_values(['f1_median'], ascending=False)
    out_path = os.path.join(args.out_root, 'reversal_param_search_wf_results.csv')
    res_df.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/raw/btc_yahoo_data.csv')
    parser.add_argument('--T_list', type=int, nargs='+', default=[7, 10, 14, 21])
    parser.add_argument('--X_list', type=float, nargs='+', default=[0.06, 0.08, 0.1, 0.12, 0.15])
    parser.add_argument('--init_train', type=int, default=500)
    parser.add_argument('--oos_window', type=int, default=63)
    parser.add_argument('--step', type=int, default=21)
    parser.add_argument('--out_root', type=str, default=os.path.join('outputs', 'backtest'))

    # A/B experiment helpers
    parser.add_argument('--feature_set', type=str, default='baseline', choices=['baseline', 'flow'])
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2])
    parser.add_argument('--baseline_summary', type=str, default='')
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--candidates_file', type=str, default='')

    args = parser.parse_args()
    main(args)
