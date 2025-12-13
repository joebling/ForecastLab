## 启动web可视化服务


python scripts/reversal_param_search_wf.py --data_path  data/raw/btc_yahoo_data.csv --T_list 7 10 14 21 --X_list 0.06 0.08 0.10 0.12 0.15 --init_train 500 --oos_window 63 --step 21

python scripts/reversal_param_search_wf.py --data_path  data/raw/btc_binance_BTCUSDT_1d.csv --T_list 7 10 14 21 --X_list 0.06 0.08 0.10 0.12 0.15 --init_train 500 --oos_window 63 --step 21


# python scripts/reversal_param_search_wf.py --T_list 7 14 21 --X_list 0.06 0.08 0.1 --init_train 500 --oos_window 63 --step 21


python -m src.data.binance_downloader --symbol BTCUSDT --interval 1d --start 2022-01-01 --end 2025-12-12 --out data/raw/btc_binance_BTCUSDT_1d.csv


python -m src.data.binance_downloader --symbol BTCUSDT --interval 1d --start 2018-01-01 --end 2025-12-12 --out data/raw/btc_binance_BTCUSDT_1d.csv




# 跑 Yahoo（示例）
python scripts/reversal_param_search_wf.py \
  --data_path data/raw/btc_yahoo_data.csv \
  --T_list 7 10 14 21 \
  --X_list 0.06 0.08 0.10 0.12 0.15 \
  --init_train 500 --oos_window 63 --step 21 \
  --out_root outputs/backtest/yahoo_2022_2025

# 跑 Binance（示例，建议 init_train 调大）
python scripts/reversal_param_search_wf.py \
  --data_path data/raw/btc_binance_BTCUSDT_1d.csv \
  --T_list 7 10 14 21 \
  --X_list 0.06 0.08 0.10 0.12 0.15 \
  --init_train 1500 --oos_window 63 --step 21 \
  --out_root outputs/backtest/binance_2018_2025

## 3) 两阶段跑法（建议这样做，更省时间且公平）

> 目标：先用 **baseline 特征** 全量搜索挑出 Top-K candidate，然后用 **flow（资金面）特征** 只复跑同一组 candidate，做到 A/B 对照且节省时间。

### Stage 1：baseline 全量搜索（得到 Top-K）

```bash
python scripts/reversal_param_search_wf.py \
  --data_path data/raw/btc_binance_BTCUSDT_1d.csv \
  --feature_set baseline \
  --stage 1 \
  --T_list 7 10 14 21 \
  --X_list 0.06 0.08 0.10 0.12 0.15 \
  --init_train 1500 --oos_window 63 --step 21 \
  --out_root outputs/backtest/binance_baseline_2018_2025
```

Stage 1 输出：
- `outputs/backtest/binance_baseline_2018_2025/reversal_param_search_wf_results.csv`
- `outputs/backtest/binance_baseline_2018_2025/meta.json`

### Stage 2：flow 只复跑 Top-K candidate（A/B 对照）

```bash
python scripts/reversal_param_search_wf.py \
  --data_path data/raw/btc_binance_BTCUSDT_1d.csv \
  --feature_set flow \
  --stage 2 \
  --baseline_summary outputs/backtest/binance_baseline_2018_2025/reversal_param_search_wf_results.csv \
  --top_k 10 \
  --init_train 1500 --oos_window 63 --step 21 \
  --out_root outputs/backtest/binance_flow_top10_2018_2025
```

> 说明：Stage 2 会从 baseline 的 summary 里取 Top-K 的 `candidate` 名称，只对这些候选产出 fold_metrics 与 summary。

（可选）如果你希望手工指定 candidate 列表，也可以准备一个文本文件（每行一个 candidate，如 `T14_X10`），并用：
- `--candidates_file path/to/candidates.txt`

### Web 可视化
- 直接启动 Streamlit 后，在左侧选择不同 run 目录（baseline vs flow）即可对照。
- Web 会读取每个 run 下的 `meta.json` 来展示 data_path / feature_set / 模型信息，并用该数据文件做“数据与标签”演示。






