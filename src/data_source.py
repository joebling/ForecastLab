"""
真实数据获取模块
支持从多个数据源获取BTC历史价格数据
"""
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

class DataSource:
    """真实数据源获取器"""
    
    def __init__(self):
        if HAS_REQUESTS:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        else:
            print("⚠ 警告: requests库未安装，网络数据获取功能受限")
    
    def get_binance_data(self, symbol: str = 'BTCUSDT', interval: str = '1d', 
                        start_date: str = '2020-01-01', end_date: str = None, 
                        limit: int = 1000) -> pd.DataFrame:
        """
        从Binance获取历史K线数据
        
        Args:
            symbol: 交易对，如'BTCUSDT'
            interval: 时间间隔，如'1d', '4h', '1h'
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'，默认为今天
            limit: 每次请求的最大数量（最大1000）
        
        Returns:
            包含OHLCV数据的DataFrame
        """
        if not HAS_REQUESTS:
            print("⚠ 需要安装requests库: pip install requests")
            return pd.DataFrame()
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 转换时间戳
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        base_url = 'https://api.binance.com/api/v3/klines'
        all_data = []
        
        current_timestamp = start_timestamp
        
        print(f"正在从Binance获取 {symbol} 数据...")
        print(f"时间范围: {start_date} 到 {end_date}")
        print(f"时间间隔: {interval}")
        
        while current_timestamp < end_timestamp:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_timestamp,
                'endTime': end_timestamp,
                'limit': limit
            }
            
            try:
                response = self.session.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # 更新时间戳到最后一条记录之后
                current_timestamp = data[-1][6] + 1  # Close time + 1ms
                
                print(f"已获取 {len(all_data)} 条记录...")
                time.sleep(0.1)  # 避免频率限制
                
            except requests.RequestException as e:
                print(f"请求错误: {e}")
                break
            except Exception as e:
                print(f"数据解析错误: {e}")
                break
        
        if not all_data:
            raise ValueError("未获取到任何数据")
        
        # 转换为DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        
        # 数据类型转换
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # 只保留需要的列
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
        
        print(f"✓ 成功获取 {len(df)} 条 {symbol} 数据")
        return df
    
    def get_coinapi_data(self, symbol: str = 'BTC', quote: str = 'USD', 
                        start_date: str = '2020-01-01', end_date: str = None,
                        period: str = '1DAY', api_key: str = None) -> pd.DataFrame:
        """
        从CoinAPI获取历史数据（需要API密钥）
        
        Args:
            symbol: 基础货币，如'BTC'
            quote: 计价货币，如'USD'
            start_date: 开始日期
            end_date: 结束日期
            period: 时间周期，如'1DAY', '1HRS', '1MIN'
            api_key: CoinAPI的API密钥
        
        Returns:
            包含OHLCV数据的DataFrame
        """
        if api_key is None:
            print("⚠ 需要CoinAPI密钥，跳过CoinAPI数据源")
            return pd.DataFrame()
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        url = f'https://rest.coinapi.io/v1/ohlcv/{symbol}/{quote}/history'
        
        params = {
            'period_id': period,
            'time_start': f'{start_date}T00:00:00',
            'time_end': f'{end_date}T23:59:59'
        }
        
        headers = {'X-CoinAPI-Key': api_key}
        
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['time_period_start'])
            df = df.rename(columns={
                'price_open': 'open',
                'price_high': 'high', 
                'price_low': 'low',
                'price_close': 'close',
                'volume_traded': 'volume'
            })
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
            
            print(f"✓ 成功从CoinAPI获取 {len(df)} 条数据")
            return df
            
        except Exception as e:
            print(f"CoinAPI获取失败: {e}")
            return pd.DataFrame()
    
    def get_yahoo_finance_data(self, symbol: str = 'BTC-USD', 
                              start_date: str = '2020-01-01', end_date: str = None) -> pd.DataFrame:
        """
        从Yahoo Finance获取数据（通过yfinance库）
        
        Args:
            symbol: 交易对，如'BTC-USD'
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            包含OHLCV数据的DataFrame
        """
        if not HAS_YFINANCE:
            print("⚠ 需要安装yfinance: pip install yfinance")
            return pd.DataFrame()
        
        try:
            # yfinance已在模块顶部导入
            pass
        except ImportError:
            print("⚠ 需要安装yfinance: pip install yfinance")
            return pd.DataFrame()
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"未获取到 {symbol} 的数据")
                return pd.DataFrame()
            
            # 重置索引并重命名列
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            print(f"✓ 成功从Yahoo Finance获取 {len(df)} 条 {symbol} 数据")
            return df
            
        except Exception as e:
            print(f"Yahoo Finance获取失败: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, source: str = 'binance', **kwargs) -> pd.DataFrame:
        """
        统一接口获取历史数据
        
        Args:
            source: 数据源 ('binance', 'coinapi', 'yahoo')
            **kwargs: 其他参数
        
        Returns:
            包含OHLCV数据的DataFrame
        """
        if source == 'binance':
            return self.get_binance_data(**kwargs)
        elif source == 'coinapi':
            return self.get_coinapi_data(**kwargs)
        elif source == 'yahoo':
            return self.get_yahoo_finance_data(**kwargs)
        else:
            raise ValueError(f"不支持的数据源: {source}")
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        保存数据到CSV文件
        
        Args:
            df: 数据DataFrame
            filename: 文件名，默认自动生成
        
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"btc_historical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df.to_csv(filename, index=False)
        print(f"✓ 数据已保存到: {filename}")
        return filename
    
    def load_or_fetch_data(self, filename: str = 'btc_historical_data.csv', 
                          source: str = 'binance', **kwargs) -> pd.DataFrame:
        """
        优先加载本地数据，如果不存在则从网络获取
        
        Args:
            filename: 本地文件名
            source: 数据源
            **kwargs: 获取数据的参数
        
        Returns:
            包含OHLCV数据的DataFrame
        """
        if os.path.exists(filename):
            print(f"发现本地数据文件: {filename}")
            try:
                df = pd.read_csv(filename)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"✓ 成功加载本地数据: {len(df)} 条记录")
                print(f"数据时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
                return df
            except Exception as e:
                print(f"加载本地数据失败: {e}")
        
        print("获取网络数据...")
        df = self.get_historical_data(source, **kwargs)
        
        if not df.empty:
            self.save_data(df, filename)
        
        return df


def demo_data_fetching():
    """演示数据获取功能"""
    print("="*60)
    print("BTC历史数据获取演示")
    print("="*60)
    
    data_source = DataSource()
    
    # 尝试多个数据源
    sources_config = [
        {
            'name': 'Binance',
            'source': 'binance',
            'params': {
                'symbol': 'BTCUSDT',
                'interval': '1d',
                'start_date': '2023-01-01',
                'limit': 365
            }
        },
        {
            'name': 'Yahoo Finance', 
            'source': 'yahoo',
            'params': {
                'symbol': 'BTC-USD',
                'start_date': '2023-01-01'
            }
        }
    ]
    
    for config in sources_config:
        print(f"\n尝试从 {config['name']} 获取数据...")
        try:
            df = data_source.get_historical_data(
                source=config['source'],
                **config['params']
            )
            
            if not df.empty:
                print(f"✓ 成功获取 {len(df)} 条数据")
                print("数据预览:")
                print(df.head())
                
                # 保存数据
                filename = f"btc_data_{config['source']}.csv"
                data_source.save_data(df, filename)
                break
                
        except Exception as e:
            print(f"✗ {config['name']} 获取失败: {e}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    demo_data_fetching()
