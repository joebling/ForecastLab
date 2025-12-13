# ForecastLab 项目重组建议

## 🗂️ 建议的目录结构

```
ForecastLab/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖列表
├── setup.py                    # 项目安装配置
├── 
├── src/                        # 📦 核心源代码
│   ├── __init__.py
│   ├── data_source.py          # 数据获取模块
│   ├── feature_engineering.py # 特征工程模块
│   ├── label_generator.py      # 标签生成模块
│   ├── model_trainer.py        # 模型训练模块
│   ├── model_evaluator.py      # 模型评估模块
│   └── predict_reversal.py     # 预测脚本
│
├── data/                       # 📊 数据文件
│   ├── raw/                    # 原始数据
│   │   ├── btc_yahoo_data.csv
│   │   └── btc_ohlcv_sample.csv
│   ├── processed/              # 处理后数据
│   │   └── processed_features_*.csv
│   └── simulated/              # 模拟数据
│       └── btc_ohlcv_sample.csv
│
├── models/                     # 🤖 训练好的模型
│   ├── production/             # 生产环境模型
│   │   └── reversal_model_latest.pkl
│   ├── experiments/            # 实验模型
│   │   └── reversal_model_*.pkl
│   └── configs/               # 模型配置
│       └── model_config.yaml
│
├── reports/                    # 📋 报告和分析
│   ├── training_reports/       # 训练报告
│   │   └── training_report_*.json
│   ├── analysis/              # 分析文档
│   │   ├── Top10_Features_Analysis.md
│   │   ├── Model_Summary_and_Prediction.md
│   │   └── Reversal_Event_Model.md
│   └── visualizations/        # 可视化图表
│       ├── feature_importance_*.png
│       ├── label_distribution.png
│       └── price_with_signals.png
│
├── outputs/                    # 🔮 预测输出
│   ├── predictions/            # 预测结果
│   │   └── prediction_result_*.json
│   └── backtest/              # 回测结果
│       └── reversal_param_search_results.csv
│
├── scripts/                    # 🛠️ 运行脚本
│   ├── main_pipeline.py        # 主训练流程
│   ├── install_requirements.py # 依赖安装脚本
│   └── run_prediction.py       # 预测运行脚本
│
├── configs/                    # ⚙️ 配置文件
│   ├── data_config.yaml        # 数据配置
│   ├── model_config.yaml       # 模型配置
│   └── feature_config.yaml     # 特征配置
│
├── tests/                      # 🧪 测试文件
│   ├── test_data_source.py
│   ├── test_feature_engineering.py
│   └── test_model_trainer.py
│
├── notebooks/                  # 📓 Jupyter笔记本
│   ├── data_exploration.ipynb
│   └── model_analysis.ipynb
│
└── lab-venv/                   # 🐍 虚拟环境
    └── ...
```

## 📋 重组执行计划

### 第一步：创建目录结构
```bash
mkdir -p src data/{raw,processed,simulated} models/{production,experiments,configs} 
mkdir -p reports/{training_reports,analysis,visualizations} outputs/{predictions,backtest}
mkdir -p scripts configs tests notebooks
```

### 第二步：移动核心源码到 src/
```bash
mv feature_engineering.py label_generator.py model_trainer.py src/
mv model_evaluator.py data_source.py predict_reversal.py src/
```

### 第三步：整理数据文件到 data/
```bash
mv btc_yahoo_data.csv data/raw/
mv btc_ohlcv_sample.csv data/simulated/
mv processed_features_*.csv data/processed/
```

### 第四步：整理模型文件到 models/
```bash
mv reversal_model_*.pkl models/experiments/
# 复制最新模型到生产环境
cp models/experiments/reversal_model_20251204_002812.pkl models/production/reversal_model_latest.pkl
```

### 第五步：整理报告到 reports/
```bash
mv training_report_*.json reports/training_reports/
mv *.md reports/analysis/
mv *.png reports/visualizations/
```

### 第六步：整理输出到 outputs/
```bash
mv prediction_result_*.json outputs/predictions/
mv reversal_param_search_results.csv outputs/backtest/
```

### 第七步：整理脚本到 scripts/
```bash
mv main_pipeline.py main_pipeline_fixed.py scripts/
mv install_requirements.py reversal_param_search.py scripts/
```

## 🔧 配置文件建议

### requirements.txt
```
pandas>=1.5.0
numpy>=1.20.0
scikit-learn>=1.2.0
lightgbm>=3.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
yfinance>=0.2.0
requests>=2.28.0
```

### README.md 大纲
```markdown
# BTC反转预测模型 (ForecastLab)

## 项目简介
统一反转预测模型，基于真实BTC历史数据训练，95%准确率

## 快速开始
1. 安装依赖: pip install -r requirements.txt
2. 训练模型: python scripts/main_pipeline.py
3. 运行预测: python src/predict_reversal.py

## 模型性能
- 准确率: 95%
- 数据: 1,433条真实BTC日线数据 (2022-2025)
- 特征: 151个技术指标

## 目录结构
[项目结构说明]
```

## 🎯 整理后的优势

1. **清晰分离**: 源码、数据、模型、报告各司其职
2. **版本管理**: 模型版本化，便于回滚和比较
3. **可复现性**: 配置文件化，环境标准化
4. **可扩展性**: 模块化设计，便于添加新功能
5. **协作友好**: 标准项目结构，便于团队协作

## 🚀 执行建议

建议分步骤执行，每步完成后验证：
1. 先创建目录结构
2. 逐步移动文件并测试
3. 更新import路径
4. 创建配置文件
5. 编写README和文档

需要我帮你执行这个重组计划吗？
