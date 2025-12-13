"""
模型评估和可视化工具
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import json

class ModelEvaluator:
    """模型评估和可视化工具"""
    
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_feature_importance(self, importance_dict: Dict, top_n: int = 20, title: str = "Feature Importance"):
        """绘制特征重要性图"""
        # 排序并取前top_n
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features, scores = zip(*sorted_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'feature_importance_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_label_distribution(self, df: pd.DataFrame, label_col: str = 'enhanced_label'):
        """绘制标签分布图"""
        plt.figure(figsize=(12, 5))
        
        # 子图1: 标签分布饼图
        plt.subplot(1, 2, 1)
        label_counts = df[label_col].value_counts()
        label_names = {-1: 'Top Events', 0: 'Neutral', 1: 'Bottom Events'}
        labels = [label_names.get(idx, f'Label_{idx}') for idx in label_counts.index]
        colors = ['red', 'gray', 'green']
        
        plt.pie(label_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Label Distribution')
        
        # 子图2: 时间序列标签分布
        plt.subplot(1, 2, 2)
        if 'timestamp' in df.columns:
            df_plot = df.copy()
            df_plot['month'] = pd.to_datetime(df_plot['timestamp']).dt.to_period('M')
            monthly_dist = df_plot.groupby(['month', label_col]).size().unstack(fill_value=0)
            
            monthly_dist.plot(kind='bar', stacked=True, color=colors, ax=plt.gca())
            plt.title('Monthly Label Distribution')
            plt.xlabel('Month')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.legend(['Top', 'Neutral', 'Bottom'])
        
        plt.tight_layout()
        plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_price_with_signals(self, df: pd.DataFrame, label_col: str = 'enhanced_label'):
        """绘制价格走势与反转信号"""
        plt.figure(figsize=(15, 8))
        
        # 价格走势
        plt.plot(df.index, df['close'], label='Price', linewidth=1, color='black', alpha=0.7)
        
        # 标记反转点
        top_points = df[df[label_col] == -1]
        bottom_points = df[df[label_col] == 1]
        
        plt.scatter(top_points.index, top_points['close'], 
                   color='red', marker='v', s=50, label='Top Events', alpha=0.8)
        plt.scatter(bottom_points.index, bottom_points['close'], 
                   color='green', marker='^', s=50, label='Bottom Events', alpha=0.8)
        
        plt.title('Price Chart with Reversal Signals')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('price_with_signals.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_future_returns_distribution(self, df: pd.DataFrame):
        """绘制未来收益分布"""
        plt.figure(figsize=(12, 5))
        
        # 子图1: 回撤分布
        plt.subplot(1, 2, 1)
        plt.hist(df['future_drawdown'].dropna(), bins=50, alpha=0.7, color='red', density=True)
        plt.axvline(df['future_drawdown'].mean(), color='darkred', linestyle='--', 
                   label=f'Mean: {df["future_drawdown"].mean():.3f}')
        plt.xlabel('Future Drawdown')
        plt.ylabel('Density')
        plt.title('Future Drawdown Distribution')
        plt.legend()
        
        # 子图2: 上涨分布
        plt.subplot(1, 2, 2)
        plt.hist(df['future_upswing'].dropna(), bins=50, alpha=0.7, color='green', density=True)
        plt.axvline(df['future_upswing'].mean(), color='darkgreen', linestyle='--', 
                   label=f'Mean: {df["future_upswing"].mean():.3f}')
        plt.xlabel('Future Upswing')
        plt.ylabel('Density')
        plt.title('Future Upswing Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('future_returns_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_rsi_signals(self, df: pd.DataFrame, label_col: str = 'enhanced_label'):
        """绘制RSI与反转信号的关系"""
        plt.figure(figsize=(15, 10))
        
        # 子图1: RSI时间序列
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['rsi_14'], label='RSI', color='blue')
        plt.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        plt.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        
        top_points = df[df[label_col] == -1]
        bottom_points = df[df[label_col] == 1]
        plt.scatter(top_points.index, top_points['rsi_14'], color='red', marker='v', s=30, alpha=0.8)
        plt.scatter(bottom_points.index, bottom_points['rsi_14'], color='green', marker='^', s=30, alpha=0.8)
        
        plt.title('RSI with Reversal Signals')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: RSI分布（按标签）
        plt.subplot(3, 1, 2)
        for label_val, color, name in [(-1, 'red', 'Top'), (0, 'gray', 'Neutral'), (1, 'green', 'Bottom')]:
            data = df[df[label_col] == label_val]['rsi_14'].dropna()
            if len(data) > 0:
                plt.hist(data, bins=30, alpha=0.6, color=color, label=f'{name} Events', density=True)
        
        plt.xlabel('RSI Value')
        plt.ylabel('Density')
        plt.title('RSI Distribution by Event Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图3: MACD分布
        plt.subplot(3, 1, 3)
        for label_val, color, name in [(-1, 'red', 'Top'), (0, 'gray', 'Neutral'), (1, 'green', 'Bottom')]:
            data = df[df[label_col] == label_val]['macd_hist'].dropna()
            if len(data) > 0:
                plt.hist(data, bins=30, alpha=0.6, color=color, label=f'{name} Events', density=True)
        
        plt.xlabel('MACD Histogram')
        plt.ylabel('Density')
        plt.title('MACD Histogram Distribution by Event Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rsi_macd_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_model_report(self, model_results: Dict, df: pd.DataFrame, save_path: str = None):
        """生成综合模型报告"""
        print("\n" + "="*80)
        print("模型评估报告")
        print("="*80)
        
        # 基础信息
        print(f"\n📊 数据概览:")
        print(f"  总样本数: {len(df):,}")
        print(f"  特征数量: {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'enhanced_label']])}")
        
        if 'enhanced_label' in df.columns:
            label_dist = df['enhanced_label'].value_counts().sort_index()
            print(f"  标签分布:")
            for label_val in [-1, 0, 1]:
                count = label_dist.get(label_val, 0)
                pct = count / len(df) * 100
                name = {-1: '顶部事件', 0: '正常期间', 1: '底部事件'}[label_val]
                print(f"    {name}: {count:,} ({pct:.1f}%)")
        
        # 特征重要性
        if 'feature_importance' in model_results:
            print(f"\n🎯 前10个重要特征:")
            importance = model_results['feature_importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, score) in enumerate(sorted_features[:10], 1):
                print(f"  {i:2d}. {feature}: {score:.4f}")
        
        # 预测性能
        if 'classification_report' in model_results:
            print(f"\n📈 分类性能:")
            print(model_results['classification_report'])
        
        # 可视化
        print(f"\n📊 生成可视化图表...")
        
        # 标签分布图
        self.plot_label_distribution(df)
        
        # 价格信号图
        if 'close' in df.columns:
            self.plot_price_with_signals(df)
        
        # 未来收益分布
        if 'future_drawdown' in df.columns and 'future_upswing' in df.columns:
            self.plot_future_returns_distribution(df)
        
        # RSI分析
        if 'rsi_14' in df.columns:
            self.plot_rsi_signals(df)
        
        # 特征重要性图
        if 'feature_importance' in model_results:
            self.plot_feature_importance(model_results['feature_importance'])
        
        print(f"\n✅ 报告生成完成!")
        print("="*80)
