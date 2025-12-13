"""
BTC反转预测模型 - 实时预测脚本
加载训练好的模型，对当前市场状态进行反转预测
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

from .feature_engineering import FeatureEngine
from .data_source import DataSource
from .model_trainer import ReversalModelTrainer

class ReversalPredictor:
    """反转预测器 - 用于实时预测"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.trainer = None
        self.feature_engine = FeatureEngine()
        self.data_source = DataSource()
        
        # 如果指定了模型路径，自动加载
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        self.trainer = ReversalModelTrainer()
        self.trainer.load_model(model_path)
        print(f"✓ 模型已加载: {model_path}")
    
    def get_latest_data(self, days: int = 100) -> pd.DataFrame:
        """获取最新的BTC数据"""
        try:
            # 尝试获取最新数据
            df = self.data_source.fetch_yahoo_data(
                symbol='BTC-USD',
                start_date=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            )
            print(f"✓ 获取到 {len(df)} 条最新数据")
            return df
        except Exception as e:
            print(f"获取数据失败: {e}")
            # 如果获取失败，尝试读取本地数据
            try:
                df = pd.read_csv('btc_yahoo_data.csv')
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.tail(days)  # 取最近的数据
                print(f"✓ 使用本地数据: {len(df)} 条")
                return df
            except Exception as e2:
                print(f"读取本地数据也失败: {e2}")
                return None
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        # 使用特征工程模块处理数据
        df_features = self.feature_engine.process_all_features(df)
        return df_features
    
    def predict_current_state(self) -> dict:
        """预测当前市场状态"""
        if self.trainer is None:
            return {"error": "模型未加载"}
        
        # 获取最新数据
        df = self.get_latest_data(days=100)
        if df is None or len(df) < 50:
            return {"error": "数据不足"}
        
        # 准备特征
        df_features = self.prepare_features(df)
        
        # 取最后一条记录进行预测
        latest_row = df_features.iloc[-1:].copy()
        
        # 准备预测数据
        try:
            X, feature_cols = self.trainer.prepare_features(latest_row)
            if len(X) == 0:
                return {"error": "特征准备失败"}
            
            # 进行预测
            predictions = self.trainer.predict(X)
            
            # 解析预测结果
            pred_class = predictions['class_pred_original'][0]
            pred_probs = predictions['class_probs'][0]
            
            # 获取当前市场数据
            current_price = df['close'].iloc[-1]
            price_change_24h = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
            
            # 计算关键指标
            latest_features = {}
            key_features = ['price_position_20', 'rsi_14', 'momentum_5', 'volume_ratio', 'kdj_k']
            for feature in key_features:
                if feature in df_features.columns:
                    latest_features[feature] = df_features[feature].iloc[-1]
            
            # 生成预测结果
            result = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "current_price": float(current_price),
                "price_change_24h": float(price_change_24h),
                "prediction": {
                    "class": int(pred_class),
                    "class_name": {-1: "顶部反转", 0: "正常", 1: "底部反转"}[pred_class],
                    "probabilities": {
                        "top_prob": float(pred_probs[0]),      # 顶部概率
                        "neutral_prob": float(pred_probs[1]),   # 正常概率
                        "bottom_prob": float(pred_probs[2])     # 底部概率
                    },
                    "confidence": float(max(pred_probs))
                },
                "key_indicators": latest_features,
                "market_signal": self._generate_signal(pred_class, max(pred_probs), latest_features),
                "data_info": {
                    "data_points": len(df),
                    "latest_date": str(df['timestamp'].iloc[-1].date()),
                    "features_count": len(feature_cols)
                }
            }
            
            return result
            
        except Exception as e:
            return {"error": f"预测失败: {str(e)}"}
    
    def _generate_signal(self, pred_class: int, confidence: float, indicators: dict) -> dict:
        """生成交易信号"""
        signal_strength = "弱"
        action = "持有"
        risk_level = "中"
        
        # 基于预测类别和置信度生成信号
        if pred_class == -1:  # 顶部反转
            if confidence > 0.8:
                signal_strength = "强"
                action = "减仓/止盈"
                risk_level = "高"
            elif confidence > 0.6:
                signal_strength = "中"
                action = "谨慎持有"
                risk_level = "中高"
        elif pred_class == 1:  # 底部反转
            if confidence > 0.8:
                signal_strength = "强"
                action = "加仓/买入"
                risk_level = "低"
            elif confidence > 0.6:
                signal_strength = "中"
                action = "考虑买入"
                risk_level = "中低"
        
        # 结合关键指标调整信号
        price_pos = indicators.get('price_position_20', 0.5)
        rsi = indicators.get('rsi_14', 50)
        
        additional_signals = []
        if price_pos > 0.8 and rsi > 70:
            additional_signals.append("价格高位 + RSI超买")
        elif price_pos < 0.2 and rsi < 30:
            additional_signals.append("价格低位 + RSI超卖")
        
        return {
            "strength": signal_strength,
            "action": action,
            "risk_level": risk_level,
            "additional_signals": additional_signals
        }
    
    def print_prediction_report(self, result: dict):
        """打印预测报告"""
        if "error" in result:
            print(f"❌ 预测错误: {result['error']}")
            return
        
        print("\n" + "="*60)
        print("🤖 BTC反转预测模型 - 实时预测报告")
        print("="*60)
        
        # 基本信息
        print(f"📅 预测时间: {result['timestamp']}")
        print(f"💰 当前价格: ${result['current_price']:,.2f}")
        print(f"📈 24h涨跌: {result['price_change_24h']:+.2f}%")
        
        # 预测结果
        pred = result['prediction']
        print(f"\n🎯 预测结果: {pred['class_name']}")
        print(f"🎲 置信度: {pred['confidence']:.1%}")
        
        # 概率分布
        probs = pred['probabilities']
        print(f"\n📊 概率分布:")
        print(f"   🔴 顶部反转: {probs['top_prob']:.1%}")
        print(f"   ⚪ 正常状态: {probs['neutral_prob']:.1%}")
        print(f"   🟢 底部反转: {probs['bottom_prob']:.1%}")
        
        # 关键指标
        indicators = result['key_indicators']
        print(f"\n📋 关键指标:")
        for key, value in indicators.items():
            print(f"   {key}: {value:.3f}")
        
        # 交易信号
        signal = result['market_signal']
        print(f"\n🚦 交易信号:")
        print(f"   信号强度: {signal['strength']}")
        print(f"   建议操作: {signal['action']}")
        print(f"   风险等级: {signal['risk_level']}")
        
        if signal['additional_signals']:
            print(f"   附加信号: {', '.join(signal['additional_signals'])}")
        
        # 数据信息
        data_info = result['data_info']
        print(f"\n📊 数据信息:")
        print(f"   数据点数: {data_info['data_points']}")
        print(f"   最新日期: {data_info['latest_date']}")
        print(f"   特征数量: {data_info['features_count']}")
        
        print("="*60)

def main():
    """主函数 - 执行实时预测"""
    print("🚀 启动BTC反转预测系统...")
    
    # 查找最新的模型文件
    import glob
    model_files = glob.glob("reversal_model_*.pkl")
    if not model_files:
        print("❌ 未找到训练好的模型文件")
        return
    
    # 使用最新的模型文件
    latest_model = sorted(model_files)[-1]
    print(f"🔍 使用模型: {latest_model}")
    
    # 初始化预测器
    predictor = ReversalPredictor(latest_model)
    
    # 执行预测
    print("🔮 正在分析当前市场状态...")
    result = predictor.predict_current_state()
    
    # 输出结果
    predictor.print_prediction_report(result)
    
    # 保存预测结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f"prediction_result_{timestamp}.json"
    
    import json
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 预测结果已保存到: {result_file}")

if __name__ == "__main__":
    main()
