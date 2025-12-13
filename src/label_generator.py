"""
统一反转预测模型标签生成模块
实现顶部/底部事件的统一标签体系
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict

class LabelGenerator:
    """反转事件标签生成器"""
    
    def __init__(self, T: int = 14, X: float = 0.10):
        """
        Args:
            T: 未来观察窗口天数
            X: 反转阈值（百分比）
        """
        self.T = T
        self.X = X
    
    def generate_reversal_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成统一反转标签 y ∈ {-1, 0, +1}"""
        df = df.copy()
        
        # 计算未来最大上涨和最大回撤
        future_max = df['close'].shift(-1).rolling(self.T).max()
        future_min = df['close'].shift(-1).rolling(self.T).min()
        
        # 未来最大回撤率（顶部信号）
        future_drawdown = (df['close'] - future_min) / df['close']
        
        # 未来最大上涨率（底部信号）
        future_upswing = (future_max - df['close']) / df['close']
        
        # 生成标签
        df['future_drawdown'] = future_drawdown
        df['future_upswing'] = future_upswing
        
        # 统一标签：-1=顶部, 0=正常, +1=底部
        df['label'] = 0
        df.loc[future_drawdown >= self.X, 'label'] = -1  # 顶部反转
        df.loc[future_upswing >= self.X, 'label'] = 1    # 底部反转
        
        # 处理同时满足条件的情况（取绝对值大者）
        both_condition = (future_drawdown >= self.X) & (future_upswing >= self.X)
        df.loc[both_condition & (future_drawdown > future_upswing), 'label'] = -1
        df.loc[both_condition & (future_upswing > future_drawdown), 'label'] = 1
        
        return df
    
    def generate_structure_labels(self, df: pd.DataFrame, pivot_window: int = 5) -> pd.DataFrame:
        """生成结构性高低点标签"""
        df = df.copy()
        
        # Pivot High (结构性高点)
        df['pivot_high'] = False
        for i in range(pivot_window, len(df) - pivot_window):
            window_high = df['high'].iloc[i-pivot_window:i+pivot_window+1]
            if df['high'].iloc[i] == window_high.max():
                df.loc[df.index[i], 'pivot_high'] = True
        
        # Pivot Low (结构性低点)
        df['pivot_low'] = False
        for i in range(pivot_window, len(df) - pivot_window):
            window_low = df['low'].iloc[i-pivot_window:i+pivot_window+1]
            if df['low'].iloc[i] == window_low.min():
                df.loc[df.index[i], 'pivot_low'] = True
        
        return df
    
    def generate_position_labels(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """生成价格位置标签"""
        df = df.copy()
        
        # 计算价格在过去N天区间的位置
        rolling_min = df['close'].rolling(lookback).min()
        rolling_max = df['close'].rolling(lookback).max()
        price_position = (df['close'] - rolling_min) / (rolling_max - rolling_min)
        
        df['price_position'] = price_position
        df['is_top_range'] = price_position >= 0.8    # 位于区间顶部
        df['is_bottom_range'] = price_position <= 0.2  # 位于区间底部
        
        return df
    
    def generate_enhanced_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成增强的反转标签（结合多种条件）"""
        df = self.generate_reversal_labels(df)
        df = self.generate_structure_labels(df)
        df = self.generate_position_labels(df)
        
        # 增强标签：结合未来回报和结构性信号
        df['enhanced_label'] = df['label']
        
        # 顶部增强条件
        top_enhanced = (
            (df['label'] == -1) & 
            (df['pivot_high'] | df['is_top_range'])
        )
        
        # 底部增强条件
        bottom_enhanced = (
            (df['label'] == 1) & 
            (df['pivot_low'] | df['is_bottom_range'])
        )
        
        df['enhanced_label'] = 0
        df.loc[top_enhanced, 'enhanced_label'] = -1
        df.loc[bottom_enhanced, 'enhanced_label'] = 1
        
        return df
    
    def compute_label_statistics(self, df: pd.DataFrame) -> Dict:
        """计算标签统计信息"""
        stats = {}
        
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            total = len(df)
            
            stats['basic_labels'] = {
                'total_samples': total,
                'top_events': label_counts.get(-1, 0),
                'bottom_events': label_counts.get(1, 0),
                'neutral_events': label_counts.get(0, 0),
                'top_ratio': label_counts.get(-1, 0) / total,
                'bottom_ratio': label_counts.get(1, 0) / total,
                'neutral_ratio': label_counts.get(0, 0) / total
            }
        
        if 'enhanced_label' in df.columns:
            enhanced_counts = df['enhanced_label'].value_counts()
            
            stats['enhanced_labels'] = {
                'total_samples': total,
                'top_events': enhanced_counts.get(-1, 0),
                'bottom_events': enhanced_counts.get(1, 0),
                'neutral_events': enhanced_counts.get(0, 0),
                'top_ratio': enhanced_counts.get(-1, 0) / total,
                'bottom_ratio': enhanced_counts.get(1, 0) / total,
                'neutral_ratio': enhanced_counts.get(0, 0) / total
            }
        
        if 'future_drawdown' in df.columns and 'future_upswing' in df.columns:
            stats['future_returns'] = {
                'avg_drawdown': df['future_drawdown'].mean(),
                'avg_upswing': df['future_upswing'].mean(),
                'max_drawdown': df['future_drawdown'].max(),
                'max_upswing': df['future_upswing'].max(),
                'drawdown_std': df['future_drawdown'].std(),
                'upswing_std': df['future_upswing'].std()
            }
        
        return stats
    
    def optimize_thresholds(self, df: pd.DataFrame, T_range: list = None, X_range: list = None) -> Dict:
        """优化T和X参数"""
        if T_range is None:
            T_range = [7, 10, 14, 21]
        if X_range is None:
            X_range = [0.06, 0.08, 0.10, 0.12, 0.15]
        
        results = []
        
        for T in T_range:
            for X in X_range:
                # 临时设置参数
                original_T, original_X = self.T, self.X
                self.T, self.X = T, X
                
                # 生成标签
                df_temp = self.generate_reversal_labels(df.copy())
                stats = self.compute_label_statistics(df_temp)
                
                # 计算平衡性指标
                if 'basic_labels' in stats:
                    top_ratio = stats['basic_labels']['top_ratio']
                    bottom_ratio = stats['basic_labels']['bottom_ratio']
                    balance_score = 1 - abs(top_ratio - bottom_ratio)
                    event_coverage = top_ratio + bottom_ratio
                    
                    results.append({
                        'T': T,
                        'X': X,
                        'top_ratio': top_ratio,
                        'bottom_ratio': bottom_ratio,
                        'balance_score': balance_score,
                        'event_coverage': event_coverage,
                        'score': balance_score * event_coverage  # 综合评分
                    })
                
                # 恢复原参数
                self.T, self.X = original_T, original_X
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
