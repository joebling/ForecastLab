"""
Plot and summarize walk-forward backtest results
从回测结果表读取 top N 候选（按 sharpe_median、f1_median 排序），为每个候选生成并保存回测可视化与摘要统计。

Usage:
  python scripts/plot_backtest_summary.py --results outputs/backtest/reversal_param_search_wf_results.csv --top 3

Outputs saved to outputs/backtest/plots/{candidate}/
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')


def plot_candidate(candidate_dir: str, out_dir: str):
    eq_path = os.path.join(candidate_dir, 'equity_curves.csv')
    tl_path = os.path.join(candidate_dir, 'tradelog.csv')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(eq_path):
        print(f"  - missing equity file: {eq_path}")
        return None

    eq = pd.read_csv(eq_path)
    # forward fill and replace remaining nans
    eq = eq.ffill().fillna(1.0)


    # mean equity across folds
    mean_eq = eq.mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    # plot each fold
    for col in eq.columns:
        ax.plot(eq.index, eq[col].cumprod(), color='gray', alpha=0.2, linewidth=0.8)
    ax.plot(mean_eq.index, mean_eq.cumprod(), color='blue', linewidth=2, label='Mean equity')
    ax.set_title(os.path.basename(candidate_dir) + ' - Mean Equity (cumprod of mean returns)')
    ax.set_xlabel('Index')
    ax.set_ylabel('Equity (normalized)')
    ax.legend()
    fig_path = os.path.join(out_dir, 'equity_mean.png')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # save basic stats
    stats = {
        'folds': eq.shape[1],
        'mean_final_equity': float(mean_eq.cumprod().iloc[-1])
    }

    # tradelog plots
    if os.path.exists(tl_path):
        try:
            tl = pd.read_csv(tl_path)
            if 'pnl' in tl.columns:
                fig, ax = plt.subplots(figsize=(8,4))
                sns.histplot(tl['pnl'].dropna(), bins=30, kde=False, ax=ax)
                ax.set_title('Trade PnL Distribution')
                ax.set_xlabel('PnL')
                fig.savefig(os.path.join(out_dir, 'pnl_hist.png'), dpi=200)
                plt.close(fig)
                stats['n_trades'] = int(len(tl))
                stats['mean_pnl'] = float(tl['pnl'].mean())
                stats['median_pnl'] = float(tl['pnl'].median())
        except Exception as e:
            print(f"  - failed to read tradelog: {e}")

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='outputs/backtest/reversal_param_search_wf_results.csv')
    parser.add_argument('--top', type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print('Results file not found:', args.results)
        raise SystemExit(1)

    df = pd.read_csv(args.results)
    # sort by sharpe_median then f1_median
    df_sorted = df.sort_values(['sharpe_median', 'f1_median'], ascending=False).reset_index(drop=True)
    topn = df_sorted.head(args.top)

    summary_rows = []
    for _, row in topn.iterrows():
        cand = row['candidate']
        cand_dir = os.path.join('outputs','backtest', cand)
        out_dir = os.path.join('outputs','backtest','plots', cand)
        print('Processing plot for', cand)
        stats = plot_candidate(cand_dir, out_dir)
        if stats is not None:
            stats.update({'candidate': cand, 'T': row['T'], 'X': row['X'], 'sharpe_median': row['sharpe_median'], 'f1_median': row['f1_median']})
            summary_rows.append(stats)

    if len(summary_rows) > 0:
        pd.DataFrame(summary_rows).to_csv('outputs/backtest/plots/top_candidates_summary.csv', index=False)
        print('Saved plots and summary to outputs/backtest/plots/')
    else:
        print('No plots generated.')
