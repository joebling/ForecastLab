# ForecastLab 部署流程（本地 → Google Cloud）

目标：部署 **flow 实验组**模型，对 **`T21_X8`** 与 **`T14_X8`** 两个 candidate 每天在 **08:00 之后**运行一次预测，并将结果可视化（先本地稳定运行，再上 Google Cloud）。

> 术语
> - **candidate**：标签参数组合（例如 `T21_X8` 表示 T=21、X=0.08）。
> - **flow 实验组**：`--feature_set flow`（含 Binance 资金面字段派生特征）。
> - **run**：一次回测/训练输出目录（`outputs/backtest/<run_name>/...`）。

---

## 0. 推荐的工程化约定（先定规则）

### 0.1 训练产物与线上产物分离

- **研究/回测输出**：`outputs/backtest/<run_name>/...`
- **线上部署产物（建议新增）**：`models/production/<deploy_name>/...`
  - `model_T21_X8.pkl`
  - `model_T14_X8.pkl`
  - `meta.json`（包含 data 源、feature_set、训练截止日期、特征列、版本号）

这样做的目的：
- 回测结果目录经常变化，不适合线上依赖。
- 线上只依赖固定的 production 目录，版本可控。

### 0.2 数据源与触发时间

- 数据源：Binance 日线（`data/raw/btc_binance_BTCUSDT_1d.csv` 或在线拉取）
- 定时：每天 08:00 之后运行（注意时区，建议统一为 **UTC** 或明确使用 **Asia/Shanghai**）

---

## 1) 本地部署流程（建议按这个顺序）

### 1.1 准备 Python 环境

- 使用当前 repo 的 `requirements.txt`
- Web 可视化使用 `webapp/requirements-web.txt`

建议使用虚拟环境（你已有 `lab-venv/`，也可以复用）。

### 1.2 训练（flow）并导出最终模型

你当前的 `scripts/reversal_param_search_wf.py` 用于研究/回测。
部署前建议做一次“定版训练”流程：

1) 选定 candidate：`T21_X8`、`T14_X8`
2) 用 **flow 特征**在“截至今天”的全量数据上训练最终模型
3) 导出模型文件到 `models/production/<deploy_name>/`

推荐新增一个脚本（建议命名）：
- `scripts/train_production_models.py`

这个脚本做的事：
- 读取 Binance 数据
- 构造 `flow` 特征
- 对每个 candidate：生成标签 → 取所有可用样本训练（或设定 rolling window）
- 保存模型（`joblib`/`pickle`）
- 保存 `meta.json`（特征列、训练样本区间、候选参数、包版本）

> 说明：如果你希望“线上模型随时间滚动更新”，也可以把训练步骤放到定时任务里（每日先更新数据，再训练/或每周训练）。

### 1.3 本地预测任务（每天跑一次）

推荐新增脚本：
- `scripts/run_daily_prediction.py`

脚本职责：
- 更新/读取最新数据（Binance downloader）
- 加工 flow 特征（与训练一致）
- 加载两个模型
- 对“最新一天”产出：
  - `pred_label`（-1/0/1）
  - `pred_proba`（三类概率）
  - `asof_date`（信号日期）
- 结果落盘到：`outputs/predictions/daily/<YYYY-MM-DD>.json` 或 `.csv`

### 1.4 本地定时调度（macOS）

本地建议用两种方式之一：

**A. launchd（macOS 原生，最稳）**
- 写一个 plist，每天固定时间触发
- 触发命令：
  - 更新数据（可选）
  - 运行预测脚本

**B. cron（简单，但环境变量容易踩坑）**

> 注意：你说“每天 8 点后”，如果数据是日线，最好确保 Binance 日线已经收盘并更新。

### 1.5 可视化（本地）

你已有 Streamlit：`webapp/streamlit_app.py`
推荐扩展一个“预测看板”：
- 最近 N 天的预测信号（两个 candidate）
- 概率曲线/热力图
- 预测信号叠加价格

预测结果建议统一从 `outputs/predictions/` 读取，这样训练/回测目录变更不会影响线上预测页面。

---

## 2) Google Cloud 上线流程（稳定后再做）

你要部署的组件通常拆成两类：

1) **定时预测任务**（每天跑一次）
2) **可视化服务**（Streamlit Web）

### 2.1 GCP 选型建议

- 定时任务：
  - **Cloud Run Job**（推荐，容器化，按需启动） + **Cloud Scheduler**（触发）
  - 或 Cloud Functions（适合轻量，但依赖/文件较多时不如容器稳）

- Web 服务：
  - **Cloud Run Service**（部署 Streamlit container）

- 存储：
  - 预测结果写入 **GCS（Cloud Storage）**
  - 或写入小型数据库（BigQuery/Firestore），但第一版用 GCS 最简单

### 2.2 容器化（Docker）

建议添加：
- `Dockerfile`（分别给 prediction job / streamlit service，或用一个镜像不同 entrypoint）

镜像里包含：
- `src/`、`scripts/`、`webapp/`
- `requirements.txt` + `webapp/requirements-web.txt`
- 线上模型文件：`models/production/<deploy_name>/...`

### 2.3 预测任务在 Cloud Run Job 中运行

流程：
1) Cloud Scheduler 每天 08:xx 触发一个 HTTP 调用
2) 调用 Cloud Run Job 执行 `python scripts/run_daily_prediction.py`
3) 输出写入 GCS：
   - `gs://<bucket>/predictions/daily/<YYYY-MM-DD>.json`

### 2.4 Streamlit 在 Cloud Run Service 中运行

- Streamlit 启动时读取 GCS 上的预测结果
- 展示：最近 N 天、两个 candidate 的预测序列

### 2.5 权限与密钥

- 预测 job / Web service 使用同一个 Service Account
- 赋权：读写 GCS bucket（Storage Object Admin 或更小权限）

---

## 3) 推荐的版本管理与回滚

- 每次上线前制作一个版本目录：
  - `models/production/flow_T21X8_T14X8_YYYYMMDD/`
- `meta.json` 写入：
  - Git commit hash
  - 训练数据截止日期
  - feature_set = flow
  - candidate 列表
  - 依赖版本（pip freeze）

回滚：只需让线上任务切换到上一版本 production 目录。

---

## 4) 验收清单（建议上线前做）

### 4.1 本地验收
- [ ] 连续运行 7 天无 crash
- [ ] 预测结果每天只生成 1 份（具备幂等：重复跑不会生成多份冲突）
- [ ] 数据更新失败时能告警或至少保留日志
- [ ] Streamlit 页面能显示最新预测

### 4.2 线上验收
- [ ] Cloud Scheduler 按时触发
- [ ] Cloud Run Job 正常结束，日志可追踪
- [ ] GCS 结果文件存在
- [ ] Cloud Run Service 正常访问

---

## 5) 下一步需要你确认的点

为了把“部署脚本”落地到代码层（真正可运行），需要你确认：

1) 预测对象
- 你要预测的是“明天/未来 T 天是否发生 ±X 的反转事件”的 **标签** 吗？
- 还是要输出“概率分布 + 信号建议”？

2) 数据
- 线上数据：继续用 CSV（由 downloader 更新）还是直接在线拉 Binance API？

3) 模型更新策略
- 上线后模型是否：
  - A) 固定不变（只预测）
  - B) 每周/每月重训
  - C) 每天滚动重训

确认后，我可以继续把缺的脚本与 Streamlit 预测页面补齐。
