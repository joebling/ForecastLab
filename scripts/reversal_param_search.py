"""
Reversal Event Model 最优阈值自动求解脚本
- 读取历史K线数据（CSV）
- 回测不同T/X参数，自动寻找最佳反转阈值
- 输出各参数下的F1/召回/Sharpe等指标
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import os

# ====== 参数区 ======
DATA_PATH = "btc_ohlcv_sample.csv"  # 请放入BTC历史K线数据，包含close列
T_LIST = [7, 10, 14, 21]
X_LIST = [0.06, 0.08, 0.1, 0.12, 0.15]

# ====== 标签生成函数 ======
def compute_labels(df: pd.DataFrame, T: int, X: float) -> pd.DataFrame:
    future_max = df['close'].shift(-1).rolling(T).max()
    future_min = df['close'].shift(-1).rolling(T).min()
    future_drawdown = (df['close'] - future_min) / df['close']
    future_upswing = (future_max - df['close']) / df['close']
    df['label_top'] = (future_drawdown >= X).astype(int)
    df['label_bottom'] = (future_upswing >= X).astype(int)
    df['label'] = 0
    df.loc[df['label_top'] == 1, 'label'] = -1
    df.loc[df['label_bottom'] == 1, 'label'] = 1
    return df

# ====== 简单策略回测函数 ======
def backtest_simple(df: pd.DataFrame, T: int) -> Dict:
    # 仅做多/做空，持有T天
    returns = []
    for i in range(len(df) - T):
        if df.iloc[i]['label'] == -1:
            ret = (df.iloc[i]['close'] - df.iloc[i+T]['close']) / df.iloc[i]['close']
            returns.append(ret)
        elif df.iloc[i]['label'] == 1:
            ret = (df.iloc[i+T]['close'] - df.iloc[i]['close']) / df.iloc[i]['close']
            returns.append(ret)
    returns = np.array(returns)
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252/T)
    # F1/召回
    tp = ((df['label'] != 0) & (df['label'] == df['label'])).sum()
    recall = tp / (df['label'] != 0).sum() if (df['label'] != 0).sum() > 0 else 0
    f1 = 2 * recall / (recall + 1) if recall > 0 else 0
    return {'sharpe': sharpe, 'f1': f1, 'recall': recall, 'n': len(returns)}

# ====== 主流程 ======
def main():
    if not os.path.exists(DATA_PATH):
        print(f"请准备历史K线数据文件: {DATA_PATH}，包含close列")
        return
    df = pd.read_csv(DATA_PATH)
    results = []
    for T in T_LIST:
        for X in X_LIST:
            df2 = compute_labels(df.copy(), T, X)
            metrics = backtest_simple(df2, T)
            results.append({"T": T, "X": X, **metrics})
    res_df = pd.DataFrame(results)
    print("各参数下回测结果:")
    print(res_df.sort_values("sharpe", ascending=False))
    res_df.to_csv("reversal_param_search_results.csv", index=False)
    print("结果已保存到 reversal_param_search_results.csv")

if __name__ == "__main__":
    main()
