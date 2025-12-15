import os
import pandas as pd
from typing import List, Dict

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BACKTEST_DIR = os.path.join(BASE, 'outputs', 'backtest')
SUMMARY_CSV = os.path.join(BACKTEST_DIR, 'reversal_param_search_wf_results.csv')


def load_summary(base_dir: str = os.path.join('outputs', 'backtest')) -> pd.DataFrame:
    """Load summary CSV from a backtest run directory."""
    path = os.path.join(base_dir, 'reversal_param_search_wf_results.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def list_candidates(base_dir: str = os.path.join('outputs', 'backtest')):
    """List candidate subfolders under a backtest run directory."""
    if not os.path.exists(base_dir):
        return []
    out = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'fold_metrics.csv')):
            out.append(name)
    return out


def list_runs(root_dir: str = os.path.join('outputs', 'backtest')):
    """List available backtest run directories under outputs/backtest/* that contain a summary CSV."""
    if not os.path.exists(root_dir):
        return []
    runs = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'reversal_param_search_wf_results.csv')):
            runs.append(name)
    return runs


def load_equity(candidate: str) -> pd.DataFrame:
    path = os.path.join(BACKTEST_DIR, candidate, 'equity_curves.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_trades(candidate: str) -> pd.DataFrame:
    path = os.path.join(BACKTEST_DIR, candidate, 'tradelog.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_meta(base_dir: str = os.path.join('outputs', 'backtest')) -> Dict:
    """Load meta.json from a backtest run directory (if present)."""
    path = os.path.join(base_dir, 'meta.json')
    if not os.path.exists(path):
        return {}
    try:
        import json
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def load_fold_metrics(base_dir: str, candidate: str) -> pd.DataFrame:
    """Load per-candidate fold_metrics.csv within a run directory."""
    fm_path = os.path.join(base_dir, candidate, 'fold_metrics.csv')
    if not os.path.exists(fm_path):
        return pd.DataFrame()
    return pd.read_csv(fm_path)


def load_fold_ranges(base_dir: str) -> pd.DataFrame:
    """Load fold date ranges (if exported by the backtest script).

    Expected file: <run>/fold_ranges.csv
    Columns (recommended): fold, train_start, train_end, test_start, test_end
    """
    path = os.path.join(base_dir, 'fold_ranges.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)
