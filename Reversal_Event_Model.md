# 统一反转预测模型（Reversal Event Model）设计文档

本模型将 **顶部 → 下跌（Local Top → Downtrend）** 与 **底部 → 上涨（Local Bottom → Uptrend）** 合并为一个统一的反转预测体系，便于构建单一模型即可捕获双方向的反转概率、反转强度与未来行情风险。

---

# **1. 反转预测的核心思想（统一事件模型）**

反转事件分为：

* **顶部（Top）**：价格见顶并进入显著下跌
* **底部（Bottom）**：价格见底并进入显著上涨
* **正常（Neutral）**：无显著反转

因此模型目标：

```
y ∈ { -1, 0, +1 }
-1 = 顶部反转事件（下跌）
 0 = 正常 / 无反转
+1 = 底部反转事件（上涨）
```

在此框架下，你可以：

* 做 **单一分类模型**（推荐）
* 或两个回归模型（未来最大上涨、未来最大回撤）
* 或多任务联合学习（更强）

---

# **2. 顶部 / 底部事件统一定义（Labeling Rules）**

为了统一，底部与顶部采用对称定义体系。

## **2.1 顶部事件定义（Top Event）**

满足以下任意即可：

* 未来 T 天最大回撤 ≥ X%
* 出现结构性 pivot high
* 位于过去 N 天价格区间顶部（如 70% 分位）且未来价格下跌

## **2.2 底部事件定义（Bottom Event）**

满足以下任意即可：

* 未来 T 天最大上涨 ≥ X%
* 出现结构性 pivot low
* 位于过去 N 天价格区间底部（如 30% 分位）且未来价格上涨

## **2.3 标签统一方式**

```
y = -1 如果 future_drawdown(T) ≥ X%  且结构性顶部成立
  = +1 如果 future_upswing(T) ≥ X%   且结构性底部成立
  = 0  否则
```

---

# **3. 顶部/底部统一特征体系（Features Schema）**

所有反转信号分为四大类：趋势耗尽、资金流、链上行为、情绪宏观。

## **3.1 趋势耗尽信号（Trend Exhaustion）**

顶部：

* RSI > 70 且背离
* MACD 红柱缩短
* 价格上升乏力（更高高点 + 更低动能）
* KDJ 死叉

底部：

* RSI < 30 且背离
* MACD 绿柱缩短
* 连续下跌后出现反包
* KDJ 金叉

## **3.2 资金面信号（Market Microstructure）**

顶部：

* 做多杠杆过高（Funding Rate ↑）
* CVD 现货下降
* 稳定币流出
* ETF 净流出

底部：

* 空头累积（具备挤空条件）
* 稳定币流入
* 现货 CVD 上行
* ETF 净流入

## **3.3 链上指标（On-chain）**

顶部：

* LTH SOPR > 1 且持续获利出货
* 大户分批卖出
* NUPL 处于“贪婪区”

底部：

* MVRV < 1
* LTH SOPR < 1 后开始回升
* 旧币换手率接近完成
* 矿工难度带收缩

## **3.4 情绪与宏观（Sentiment / Macro）**

顶部：

* FGI > 80
* 交易量飙升/媒体狂热
* 宏观收紧（加息、流动性下降）

底部：

* FGI < 20
* 舆论悲观到极点
* 流动性宽松或触顶

---

# **4. 模型目标层（Labeling Target）统一方式**

统一反转模型可选择三种目标结构：

## **4.1 主目标：三分类事件预测（推荐）**

```
y ∈ {-1, 0, +1}
```

适合：

* XGBoost
* TabNet
* Transformer Encoder
* Informer / TimesFM（加分类头）

## **4.2 辅目标：回归（未来最大回撤 / 上涨）**

```
y1 = future_drawdown(T)
y2 = future_upswing(T)
```

用于：

* 多任务学习
* 提高稳健性

## **4.3 概率目标：反转概率**

```
p(top_event)   = 下跌反转概率
p(bottom_event)= 上涨反转概率
```

用于：

* 做市策略
* 风险控制
* 自适应仓位调整

---

# **5. 训练方案（Pipeline）**

1. **数据准备**

   * Kline（1h / 4h / 1d）
   * 衍生指标（RSI、MACD、布林带、动能）
   * CVD、Funding、OI、ETF 流入流出
   * 链上 MVRV、SOPR、活跃地址等

2. **标签生成（重要）**

   * 根据未来 T 天走势自动打标签
   * 保证顶部/底部比例平衡

3. **模型选择**（推荐）

   * **Tabular → LightGBM / CatBoost**（最稳）
   * **时序 → Transformer / Informer / TimesFM**（最强）
   * **强化 → PPO 做事件驱动策略**（进阶）

4. **评估指标**

   * 反转事件召回率（重点）
   * Precision / Recall（防止虚假顶部/底部）
   * 利润回测（最实际）

---

# **6. 最终统一反转预测模型输出**

模型最终输出（推荐组合）：

```
① 反转类别：{-1,0,+1}
② 顶部概率 p_top
③ 底部概率 p_bottom
④ 未来最大上涨幅度（回归）
⑤ 未来最大回撤幅度（回归）
```

你可以这样使用：

* 当 p_top ↑ 且 future_drawdown ↑ → 顶部信号强
* 当 p_bottom ↑ 且 future_upswing ↑ → 底部信号强
* 混合后用于仓位管理、入场、止盈、止损

---

如需进一步补充：

* 特征字段表（Feature Dictionary）
* 模型训练伪代码（Trainer + Dataloader）
* 反转事件可视化案例
* 反转信号强度评分

我可以继续扩展至完整工程文档。

## Feature Schema（字段名 / 含义 / 来源）

### 1. 技术指标（Technical Indicators）

| 字段名         | 含义                    | 来源      |
| ----------- | --------------------- | ------- |
| rsi_14      | 14日 RSI，相对强弱反转信号      | K线计算    |
| macd        | DIF 值，趋势衰竭参考          | K线计算    |
| macd_signal | DEA 信号线               | K线计算    |
| macd_hist   | MACD 柱体强度，顶部/底部收敛信号关键 | K线计算    |
| bb_upper    | 布林带上轨                 | K线计算    |
| bb_middle   | 布林带中轨                 | K线计算    |
| bb_lower    | 布林带下轨                 | K线计算    |
| kdj_k       | KDJ K 值               | K线计算    |
| kdj_d       | KDJ D 值               | K线计算    |
| kdj_j       | KDJ J 值，超买/超卖极值判断     | K线计算    |
| atr         | 波动率，用于判断反转爆发力度        | K线计算    |
| volume      | 成交量                   | 交易所现货K线 |

---

### 2. 资金面（Order Flow / Market Microstructure）

| 字段名                  | 含义                                | 来源                      |
| -------------------- | --------------------------------- | ----------------------- |
| funding_rate         | 永续合约资金费率，多空杠杆偏向                   | 合约API                   |
| open_interest        | 全市场未平仓合约量                         | 合约API                   |
| cvd_spot             | 现货 Cumulative Volume Delta，真实买盘强度 | Order Book / CVD 数据商    |
| cvd_perp             | 永续 CVD                            | 合约盘口                    |
| stablecoin_inflow    | 稳定币流入交易所                          | Glassnode / CryptoQuant |
| btc_exchange_outflow | BTC 流出量（囤币行为）                     | Glassnode               |
| etf_netflow          | ETF 净申购/赎回                        | ETF 披露数据                |

#### Binance 现货日K可直接补充的资金面特征（无需额外数据源）

若使用 Binance Kline（如 `BTCUSDT 1d`），除 OHLCV 外还可获得成交行为字段，可作为“资金面/微观结构”特征补充：

- `quote_asset_volume`：成交额（USDT），比 `volume`（BTC数量）更接近“资金规模/资金流强度”。
- `number_of_trades`：成交笔数（活跃度/拥挤度 proxy）。
- `taker_buy_base_asset_volume` / `taker_buy_quote_asset_volume`：主动买入成交量/成交额（buy pressure proxy）。

推荐的简洁派生特征（用于反转识别的动能衰竭/吸筹信号）：

- **主动买入占比（Buy Ratio）**：
  - `taker_buy_quote_asset_volume / quote_asset_volume`（或用 base 版本）
- **单笔平均成交额（Avg Trade Size）**：
  - `quote_asset_volume / number_of_trades`
- **量价/资金背离（Divergence）**（概念层面）：
  - 价格创新高但 `buy_ratio` 或成交额未同步走强 → 上涨动能衰竭（顶部风险）
  - 价格创新低但 `buy_ratio` 回升 → 吸筹/抄底出现（底部机会）

> 注意：特征必须严格使用 t 时刻可获得的数据（避免引入 t+1 信息）；并保证全链路“切日口径”一致（建议全程使用同数据源的日线）。

---

### 3. 链上指标（On-chain）

| 字段名               | 含义                            | 来源                    |
| ----------------- | ----------------------------- | --------------------- |
| mvrv              | Market Value / Realized Value | Glassnode             |
| lth_sopr          | 长期持有者 SOPR，反转必要指标             | Glassnode             |
| sth_sopr          | 短期持有者 SOPR                    | Glassnode             |
| active_addresses  | 活跃地址数，链上活跃度                   | Glassnode / Tokenview |
| miner_outflow     | 矿工卖压                          | Glassnode             |
| difficulty_ribbon | 难度带压缩系数                       | On-chain 计算           |

---

### 4. 情绪与宏观（Sentiment / Macro）

| 字段名              | 含义                      | 来源             |
| ---------------- | ----------------------- | -------------- |
| fgi              | 恐惧与贪婪指数（0-100）          | Alternative.me |
| google_trend_btc | Google Trend“bitcoin”指数 | Google Trend   |
| usd_liquidity    | 美元流动性（RRP + TGA）        | FRED           |
| vix              | 美股波动率指数                 | CBOE           |
| dxy              | 美元指数                    | 外汇行情           |
| rate_policy      | 当前联储政策方向（加息/降息）         | FOMC           |

---

## 标签定义（Labeling）

### 1. 顶部（y = -1）建议定义

```
future_drawdown(T) ≥ X%
```

推荐参数：

* T = 10～20 天（中期反转）
* X% = **8%～15%**（BTC 对应合理区间）

BTC 的典型顶部下跌：

* 小回调：5-8%
* 中等：10-15%
* 熊市反转：20-30%+

所以模型用于“预测反转”，采用 **10%-15%** 最稳健。

---

### 2. 底部（y = +1）建议定义

```
future_upswing(T) ≥ X%
```

推荐：

* T = 10～20 天
* X% = **8%-12%**

BTC 的典型底部反弹：

* 反抽：5-8%
* 强反弹：10%-20%

若你想更灵敏可用：

* X = 6%
  若你想更稳健可用：
* X = 10%-12%

---

### 3. 正常（y = 0）

当且仅当：

```
future_drawdown(T) < X% 且 future_upswing(T) < X%
```

表示未来没有发生明显反转。

---

如需，我可以再补充：

* 参数调优建议（如何选择最优 T / X）
* 如何用 TimesFM Transformer 进行多任务训练
* 如何合并回归任务与分类任务

---

# 可选扩展（工程实现细节）

下面把你要的五个模块一次性补齐：

1. 基于 BTC 历史数据的最优 X% 自动求解脚本（回测寻找最佳反转阈值）

2. 统一反转模型的训练 Pipeline 伪代码（含特征工程）

3. TimesFM / Transformer + TabNet 多任务 Reversal Model 设计

4. 事件强度评分（Reversal Strength Score, RSS）构建方法

5. 数据集切分策略（防止信息泄露）

---

## 1) 最优 X% 自动求解脚本（思路 + 样例代码）

目标：用历史数据回测不同 X（阈值）和 T（窗口），找到在特定评分指标下的最优阈值（例如 F1、召回或策略回测净利润）。

思路：

* 对于每对 (T, X) 在历史每个时间点打标签（top/bottom/neutral）
* 用简单策略（例：当模型/规则产生 top 信号则平仓或减仓）进行逐日回测
* 计算评价指标（F1、召回、Sharpe、MaxDrawdown）
* 选出在目标指标上最优的 (T, X)

```python
# 简化示例（pandas）
import pandas as pd, numpy as np

def compute_labels(df, T, X):
    # df must contain 'close'
    future_max = df['close'].shift(-1).rolling(T).max()
    future_min = df['close'].shift(-1).rolling(T).min()
    future_drawdown = (df['close'] - future_min) / df['close']
    future_upswing = (future_max - df['close']) / df['close']
    df['label_top'] = (future_drawdown >= X).astype(int)
    df['label_bottom'] = (future_upswing >= X).astype(int)
    return df

# 回测规则示例：当label_top==1 -> 做空单日持有T天；label_bottom->做多T天
# 评估使用净收益、Sharpe、召回等

best = []
for T in [7,10,14,21]:
    for X in [0.06,0.08,0.1,0.12,0.15]:
        df2 = compute_labels(df.copy(), T, X)
        # 这里写简单的策略回测逻辑，计算收益并记录指标
        metrics = backtest_simple(df2, T)
        best.append((T,X,metrics['f1'], metrics['sharpe']))

# 选择目标指标最大的 (T,X)
```

说明：`backtest_simple` 可实现：当 label_top 触发则在 t+1 做空并持有 T 天，收益为相应的价格变动；为防止 lookahead，标签生成需在训练集外验证。

---

## 2) 训练 Pipeline 伪代码（含特征工程）

```
# 数据加载
raw = load_klines()  # 1h/4h/1d
onchain = load_onchain()
deriv = load_derivatives()
merge_df = merge_sources(raw,onchain,deriv)

# 特征工程（Feature Factory）
features = []
features += compute_technical_features(merge_df)
features += compute_onchain_features(merge_df)
features += compute_deriv_features(merge_df)
features += compute_macro_features(merge_df)

# 滚动窗口特征（lag/rolling）
for f in features:
    df[f+'_lag1'] = df[f].shift(1)
    df[f+'_r7'] = df[f].rolling(7).mean()

# 标签生成
df = generate_labels(df, T=14, X=0.10)

# 划分数据集（见下文详细策略）
train, val, test = time_series_split(df)

# 训练 LightGBM (baseline)
lgb_train = lgb.Dataset(train[features], label=train['y'])
model = lgb.train(params, lgb_train, valid_sets=[...])

# 训练 CatBoost (可选)
# 训练 TimesFM (时序模型) -> 预测未来序列

# 集成：把 ML 的概率输出与时序预测合并
ensemble_input = concat([ml_probs, timesfm_forecast_features])
meta_model = train_meta(ensemble_input, label)
```

注意：每一步均需保存特征 pipeline（scaler, imputer）以便生产环境复现。

---

## 3) TimesFM / Transformer + TabNet 多任务设计

目标：一个模型同时完成分类（-1/0/+1）与回归（future_drawdown, future_upswing）。

### 架构建议：

* **输入**：历史价格序列（window L，如 256 日），以及对齐的时序特征（funding, mvrv, sopr 等）
* **主干**：TimesFM 或 Transformer Encoder 提取时序表示
* **任务头**：

  * 分类头（Softmax）输出三类概率
  * 回归头1：预测 future_drawdown (scalar)
  * 回归头2：预测 future_upswing (scalar)
  * 可选：辅助二分类头 p_top, p_bottom

### 损失函数（Multi-task）:

```
Loss = w_cls * CE(y_cls, p_cls) + w_dd * MSE(y_dd, pred_dd) + w_up * MSE(y_up, pred_up)
```

权重 (w_*) 可用 Val Set 调优。

### TabNet 角色：

* TabNet 可用于纯表格（非序列）特征，如链上/资金面/情绪特征
* 集成方式：TimesFM 提取序列 embedding，TabNet 处理静态表格 embedding，二者 concat 后送任务头

---

## 4) 事件强度评分 RSS（Reversal Strength Score）构建方法

目标：把多任务输出转为单一可用的强度分值（0-100）。

方案示例：

```
RSS = sigmoid(a*(p_top - p_bottom) + b*pred_drawdown_norm - c*pred_upswing_norm)
RSS_scaled = RSS * 100
```

其中：

* p_top, p_bottom 为分类输出概率
* pred_drawdown_norm = pred_drawdown / max_expected_drawdown (归一化)
* pred_upswing_norm 类似
* a,b,c 为可训练或手动调节系数

或者直接用学习的 meta 模型：

* 把 [p_top,p_bottom,pred_dd,pred_up, technical_flags] 作为输入，训练一个小的 LightGBM 输出 RSS（回归，0-100）

RSS 用途：

* 风险分级（>=80 极高；60-80 高；30-60 警戒；<30 安全）

---

## 5) 数据集切分策略（防止信息泄露）

时间序列数据必须小心切分，避免未来数据泄露到训练中。推荐策略：

1. **时间块划分（Time-based Holdout）**

   * 按时间切分 Train / Val / Test（例如 70% / 15% / 15%），保证时间顺序

2. **滑动窗口交叉验证（Rolling CV）**

   * 多次滚动扩展训练集，并在随后若干天进行验证，测试在更远窗口

3. **特征滞后与窗口化**

   * 所有特征仅使用 t-1 及之前数据生成（严格 shift），避免未来泄露

4. **事件周围剔除（Buffering）**

   * 在训练/验证边界附近留一个 buffer（如 T 天）避免标签产生跨区影响

5. **避免泄露的工程要点**

   * 不要在特征中使用未来已知的 onchain 聚合统计（如未来 rolling mean 未 shift）
   * 生产化时保存训练时的 scaler/encoder
