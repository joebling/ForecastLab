# BTC反转预测模型 (ForecastLab)

## 项目简介
统一反转预测模型，基于真实BTC历史数据训练，使用机器学习技术预测比特币价格的反转事件（顶部和底部）。

## 模型性能
- **准确率**: 95%
- **数据源**: 1,433条真实BTC日线数据 (2022-2025)
- **特征数量**: 151个技术指标
- **模型类型**: LightGBM分类器

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 训练模型
```bash
python scripts/main_pipeline.py
```

### 3. 运行预测
```bash
python src/predict_reversal.py
```

## 项目结构

```
ForecastLab/
├── src/                        # 核心源代码
│   ├── data_source.py          # 数据获取模块
│   ├── feature_engineering.py # 特征工程模块
│   ├── label_generator.py      # 标签生成模块
│   ├── model_trainer.py        # 模型训练模块
│   ├── model_evaluator.py      # 模型评估模块
│   └── predict_reversal.py     # 预测脚本
├── data/                       # 数据文件
│   ├── raw/                    # 原始数据
│   ├── processed/              # 处理后数据
│   └── simulated/              # 模拟数据
├── models/                     # 训练好的模型
│   ├── production/             # 生产环境模型
│   └── experiments/            # 实验模型
├── reports/                    # 报告和分析
│   ├── analysis/              # 分析文档
│   └── visualizations/        # 可视化图表
├── outputs/                    # 预测输出
│   ├── predictions/            # 预测结果
│   └── backtest/              # 回测结果
├── scripts/                    # 运行脚本
├── configs/                    # 配置文件
└── tests/                      # 测试文件
```

## 核心功能

### 特征工程
- 151个技术指标
- 价格动量指标
- 成交量分析
- 波动性指标
- 相对强弱指数

### 标签生成
- 三分类标签系统：
  - 0: 顶部反转事件
  - 1: 正常期间
  - 2: 底部反转事件

### 模型训练
- LightGBM分类器
- 交叉验证
- 特征重要性分析
- 模型性能评估

## 使用说明

详细的使用说明和技术文档请查看 `reports/analysis/` 目录下的相关文档：
- `Model_Summary_and_Prediction.md` - 模型总结和预测结果
- `Top10_Features_Analysis.md` - 重要特征分析

## 技术栈
- Python 3.8+
- pandas, numpy - 数据处理
- scikit-learn - 机器学习
- lightgbm - 梯度提升
- matplotlib, seaborn - 数据可视化
- yfinance - 金融数据获取

## 许可证
本项目仅供学习和研究使用。

## 联系方式
ForecastLab Team
