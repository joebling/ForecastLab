"""
反转预测模型特征工程模块
包含技术指标、资金面、链上指标、情绪与宏观特征的计算
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureEngine:
    """统一反转预测特征工程引擎"""
    
    def __init__(self):
        self.feature_names = []
    
    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标特征"""
        df = df.copy()
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        # 布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # KDJ
        df['kdj_k'], df['kdj_d'], df['kdj_j'] = self._calculate_kdj(df['high'], df['low'], df['close'])
        
        # ATR
        df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])
        
        # 动能指标
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # 价格位置
        df['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min())
        
        return df
    
    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量相关特征"""
        df = df.copy()
        
        # 成交量移动平均
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ma_30'] = df['volume'].rolling(30).mean()
        
        # 成交量比率
        df['volume_ratio'] = df['volume'] / df['volume_ma_30']
        
        # 价量背离
        df['price_volume_divergence'] = (df['close'].pct_change() * df['volume'].pct_change()).rolling(5).mean()
        
        return df
    
    def compute_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市场结构特征（模拟资金面数据）"""
        df = df.copy()
        
        # 模拟资金费率（基于价格动量）
        df['funding_rate'] = df['close'].pct_change().rolling(24).mean() * 100
        
        # 模拟未平仓合约（基于成交量）
        df['open_interest'] = df['volume'].rolling(24).sum()
        
        # 模拟CVD（累积成交量增量）
        df['cvd_spot'] = (df['volume'] * np.where(df['close'] > df['open'], 1, -1)).cumsum()
        
        # 模拟稳定币流入（基于价格变化）
        df['stablecoin_inflow'] = -df['close'].pct_change().rolling(7).sum() * df['volume'].rolling(7).mean()
        
        return df
    
    def compute_onchain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算链上指标特征（模拟数据）"""
        df = df.copy()
        
        # 模拟MVRV（基于价格历史）
        realized_price = df['close'].expanding().mean()
        df['mvrv'] = df['close'] / realized_price
        
        # 模拟SOPR（基于价格变化）
        df['lth_sopr'] = df['close'] / df['close'].shift(180)  # 长期持有者
        df['sth_sopr'] = df['close'] / df['close'].shift(30)   # 短期持有者
        
        # 模拟活跃地址数
        df['active_addresses'] = df['volume'].rolling(7).mean() + np.random.normal(0, 0.1, len(df))
        
        # 模拟矿工流出
        df['miner_outflow'] = df['close'].pct_change().rolling(30).std() * df['volume']
        
        return df
    
    def compute_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算情绪与宏观特征（模拟数据）"""
        df = df.copy()
        
        # 模拟恐惧贪婪指数
        price_momentum = df['close'].pct_change(30)
        volatility = df['close'].pct_change().rolling(30).std()
        df['fgi'] = 50 + price_momentum * 100 - volatility * 200
        df['fgi'] = df['fgi'].clip(0, 100)
        
        # 模拟Google趋势
        df['google_trend_btc'] = df['volume'].rolling(7).mean() / df['volume'].rolling(30).mean() * 50
        
        # 模拟宏观指标
        df['usd_liquidity'] = 100 - df['close'].pct_change().rolling(90).std() * 1000
        df['vix'] = volatility * 100
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, features: list, lags: list = [1, 2, 3, 7]) -> pd.DataFrame:
        """添加滞后特征"""
        df = df.copy()
        for feature in features:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, features: list, windows: list = [3, 7, 14, 30]) -> pd.DataFrame:
        """添加滚动窗口特征"""
        df = df.copy()
        for feature in features:
            if feature in df.columns:
                for window in windows:
                    df[f'{feature}_ma{window}'] = df[feature].rolling(window).mean()
                    df[f'{feature}_std{window}'] = df[feature].rolling(window).std()
        return df
    
    def process_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理所有特征"""
        # 基础特征
        df = self.compute_technical_features(df)
        df = self.compute_volume_features(df)
        df = self.compute_market_structure_features(df)
        df = self.compute_onchain_features(df)
        df = self.compute_sentiment_features(df)
        
        # 核心特征列表
        core_features = ['rsi_14', 'macd', 'macd_hist', 'atr', 'volume_ratio', 
                        'funding_rate', 'mvrv', 'lth_sopr', 'sth_sopr', 'fgi']
        
        # 添加滞后和滚动特征
        df = self.add_lag_features(df, core_features)
        df = self.add_rolling_features(df, core_features)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """计算MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """计算布林带"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def _calculate_kdj(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9):
        """计算KDJ"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        k = rsv.ewm(alpha=1/3).mean()
        d = k.ewm(alpha=1/3).mean()
        j = 3 * k - 2 * d
        
        return k, d, j
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """计算ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
