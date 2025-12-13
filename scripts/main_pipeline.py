"""
统一反转预测模型主运行脚本（修复版）
整合特征工程、标签生成、模型训练的完整流程
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import json

# 添加上级目录到路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from src.feature_engineering import FeatureEngine
from src.label_generator import LabelGenerator
from src.model_trainer import ReversalModelTrainer
from src.model_evaluator import ModelEvaluator
from src.data_source import DataSource

def generate_sample_data(n_samples: int = 1000, start_price: float = 50000) -> pd.DataFrame:
    """生成模拟BTC价格数据"""
    np.random.seed(42)
    
    # 生成价格时间序列
    returns = np.random.normal(0.001, 0.02, n_samples)  # 日收益率
    prices = [start_price]
    
    for i in range(n_samples - 1):
        # 添加一些趋势和波动性
        trend = 0.0005 * np.sin(i / 100)  # 长期趋势
        volatility = 0.015 + 0.01 * np.sin(i / 50)  # 变化的波动性
        
        daily_return = returns[i] + trend + np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    # 生成OHLCV数据
    data = []
    for i, close_price in enumerate(prices):
        # 生成合理的OHLC数据
        daily_range = close_price * np.random.uniform(0.01, 0.04)
        high = close_price + np.random.uniform(0, daily_range * 0.7)
        low = close_price - np.random.uniform(0, daily_range * 0.7)
        open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def main():
    """主运行函数"""
    print("="*60)
    print("统一反转预测模型 - 完整训练流程（修复版）")
    print("="*60)
    
    # 1. 数据准备
    print("\n1. 准备数据...")
    
    # 初始化数据源
    data_source = DataSource()
    
    # 尝试获取真实数据
    real_data_sources = [
        {
            'name': 'Binance',
            'source': 'binance',
            'filename': 'btc_binance_data.csv',
            'params': {
                'symbol': 'BTCUSDT',
                'interval': '1d',
                'start_date': '2022-01-01',
                'limit': 1000
            }
        },
        {
            'name': 'Yahoo Finance',
            'source': 'yahoo', 
            'filename': 'data/raw/btc_yahoo_data.csv',
            'params': {
                'symbol': 'BTC-USD',
                'start_date': '2022-01-01'
            }
        }
    ]
    
    df = None
    data_source_used = "模拟数据"
    
    # 尝试从真实数据源获取
    for source_config in real_data_sources:
        try:
            print(f"尝试从{source_config['name']}获取数据...")
            df = data_source.load_or_fetch_data(
                filename=source_config['filename'],
                source=source_config['source'],
                **source_config['params']
            )
            
            if not df.empty and len(df) > 100:  # 确保数据量足够
                data_source_used = source_config['name']
                print(f"✓ 成功从{source_config['name']}获取数据")
                break
            else:
                print(f"⚠ {source_config['name']}数据不足，尝试下一个数据源")
                df = None
                
        except Exception as e:
            print(f"✗ {source_config['name']}获取失败: {e}")
            continue
    
    # 如果真实数据获取失败，使用模拟数据
    if df is None or df.empty:
        print("所有真实数据源获取失败，使用模拟数据...")
        df = generate_sample_data(n_samples=1000)
        data_source_used = "模拟数据"
    
    print(f"✓ 数据源: {data_source_used}")
    print(f"数据样本数: {len(df)}")
    print(f"数据时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 数据质量检查
    print(f"数据质量检查:")
    print(f"  缺失值: {df.isnull().sum().sum()}")
    print(f"  价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"  平均日成交量: {df['volume'].mean():.0f}")
    
    # 2. 特征工程
    print("\n2. 特征工程...")
    feature_engine = FeatureEngine()
    df_features = feature_engine.process_all_features(df)
    
    print(f"生成特征数: {len([col for col in df_features.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])}")
    
    # 3. 标签生成
    print("\n3. 生成标签...")
    
    # 优化参数
    label_generator = LabelGenerator()
    optimization_results = label_generator.optimize_thresholds(df_features)
    
    print("最优参数组合:")
    for i, result in enumerate(optimization_results[:3]):
        print(f"  {i+1}. T={result['T']}, X={result['X']:.2f}, "
              f"Score={result['score']:.3f}, "
              f"Top={result['top_ratio']:.2%}, Bottom={result['bottom_ratio']:.2%}")
    
    # 使用最优参数
    best_params = optimization_results[0]
    label_generator.T = best_params['T']
    label_generator.X = best_params['X']
    
    # 生成最终标签
    df_labeled = label_generator.generate_enhanced_labels(df_features)
    
    # 统计信息
    stats = label_generator.compute_label_statistics(df_labeled)
    print(f"\n标签统计 (T={label_generator.T}, X={label_generator.X}):")
    if 'enhanced_labels' in stats:
        enhanced_stats = stats['enhanced_labels']
        print(f"  顶部事件: {enhanced_stats['top_events']} ({enhanced_stats['top_ratio']:.2%})")
        print(f"  底部事件: {enhanced_stats['bottom_events']} ({enhanced_stats['bottom_ratio']:.2%})")
        print(f"  正常期间: {enhanced_stats['neutral_events']} ({enhanced_stats['neutral_ratio']:.2%})")
    
    # 4. 模型训练
    print("\n4. 模型训练...")
    
    # 使用增强标签训练
    df_clean = df_labeled.copy()
    df_clean['label'] = df_clean['enhanced_label']
    
    # 训练LightGBM模型
    print("训练LightGBM分类模型...")
    trainer = ReversalModelTrainer(model_type='lightgbm')
    
    try:
        results = trainer.train(df_clean, validation_split=0.2)
        print("✓ 模型训练完成")
        
        # 打印结果
        if 'classification_report' in results:
            print("\n分类报告:")
            print(results['classification_report'])
        
        # 特征重要性
        if 'feature_importance' in results:
            print("\n前10个重要特征:")
            importance = results['feature_importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feature, score in sorted_features[:10]:
                print(f"  {feature}: {score:.0f}")
        
        # 保存模型
        model_path = f"reversal_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        trainer.save_model(model_path)
        print(f"\n✓ 模型已保存到: {model_path}")
        
        # 5. 模型评估和可视化
        print("\n5. 模型评估和可视化...")
        try:
            from src.model_evaluator import ModelEvaluator
            evaluator = ModelEvaluator()
            
            # 绘制标签分布
            print("生成标签分布图...")
            evaluator.plot_label_distribution(df_labeled, label_col='enhanced_label')
            
            # 绘制特征重要性
            if 'feature_importance' in results:
                print("生成特征重要性图...")
                evaluator.plot_feature_importance(results['feature_importance'], top_n=15, 
                                                title="Reversal Model Feature Importance")
            
            print("✓ 可视化完成")
        except Exception as viz_error:
            print(f"可视化模块出错: {viz_error}")
        
    except Exception as e:
        print(f"⚠ 模型训练出错: {e}")
        print("使用简化版本进行演示...")
        
        # 简化版本的模型训练演示
        from collections import Counter
        
        try:
            # 准备数据
            X, feature_cols = trainer.prepare_features(df_clean)
            y = trainer.prepare_labels(df_clean, 'label')
            
            # 基本统计
            label_dist = Counter(y)
            print(f"标签分布: {dict(label_dist)}")
            
            print("✓ 数据准备完成")
        except Exception as prep_error:
            print(f"数据准备失败: {prep_error}")
    
    # 6. 生成报告
    print("\n6. 生成报告...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_source': data_source_used,
        'data_info': {
            'total_samples': len(df),
            'feature_count': len([col for col in df_features.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]),
            'time_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max())
            },
            'price_range': {
                'min': float(df['close'].min()),
                'max': float(df['close'].max()),
                'mean': float(df['close'].mean())
            }
        },
        'optimal_params': best_params,
        'label_stats': stats,
        'model_info': {
            'model_type': trainer.model_type,
            'feature_names': trainer.feature_names
        }
    }
    
    # 保存报告
    report_path = f"data/processed/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ 训练报告已保存到: {report_path}")
    
    # 7. 数据导出
    print("\n7. 导出处理后数据...")
    
    # 导出特征数据
    feature_data_path = f"data/processed/processed_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_labeled.to_csv(feature_data_path, index=False)
    print(f"✓ 特征数据已保存到: {feature_data_path}")
    
    print("\n" + "="*60)
    print("训练流程完成！")
    print("="*60)
    
    return df_labeled, trainer, report

if __name__ == "__main__":
    df, trainer, report = main()
