"""
统一反转预测模型训练模块
支持多种模型（LightGBM、CatBoost、多任务学习）
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

class ReversalModelTrainer:
    """统一反转预测模型训练器"""
    
    def __init__(self, model_type: str = 'lightgbm', random_state: int = 42):
        """
        Args:
            model_type: 模型类型 ('lightgbm', 'catboost', 'multitask')
            random_state: 随机种子
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_mapping = {-1: 0, 0: 1, 1: 2}  # 映射到0,1,2用于分类
        self.reverse_mapping = {0: -1, 1: 0, 2: 1}
        
    def prepare_features(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """准备特征数据"""
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'enhanced_label',
                          'future_drawdown', 'future_upswing', 'pivot_high', 'pivot_low',
                          'is_top_range', 'is_bottom_range', 'price_position']
        
        # 选择特征列，排除非数值列
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols and not col.startswith('Unnamed'):
                # 检查是否为数值类型
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']:
                    feature_cols.append(col)
        
        # 处理缺失值
        df_features = df[feature_cols].fillna(method='ffill').fillna(0)
        
        # 确保所有特征都是数值类型
        for col in feature_cols:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)
        
        self.feature_names = feature_cols
        return df_features.values, feature_cols
    
    def prepare_labels(self, df: pd.DataFrame, label_col: str = 'label') -> np.ndarray:
        """准备标签数据"""
        labels = df[label_col].fillna(0)
        # 映射标签到0,1,2
        mapped_labels = labels.map(self.label_mapping)
        return mapped_labels.values
    
    def train_lightgbm_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                                X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """训练LightGBM分类模型"""
        if not HAS_LGB:
            raise ImportError("LightGBM not installed. Please install with: pip install lightgbm")
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )
        
        # 特征重要性
        feature_importance = dict(zip(self.feature_names, self.model.feature_importance()))
        
        return {'feature_importance': feature_importance}
    
    def train_catboost_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """训练CatBoost分类模型"""
        if not HAS_CATBOOST:
            raise ImportError("CatBoost not installed. Please install with: pip install catboost")
        
        self.model = cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_state=self.random_state,
            verbose=100,
            early_stopping_rounds=100
        )
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=100
        )
        
        # 特征重要性
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return {'feature_importance': feature_importance}
    
    def train_multitask_model(self, df: pd.DataFrame) -> Dict:
        """训练多任务模型（分类+回归）"""
        # 这里实现一个简单的多任务模型框架
        # 实际应用中可以使用更复杂的神经网络架构
        
        X, feature_cols = self.prepare_features(df)
        y_cls = self.prepare_labels(df, 'label')
        y_reg_dd = df['future_drawdown'].fillna(0).values
        y_reg_up = df['future_upswing'].fillna(0).values
        
        # 标准化特征
        if HAS_SKLEARN:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # 简单的多任务模型：分别训练分类器和回归器
        results = {}
        
        # 分类任务
        if HAS_LGB:
            # 分类模型
            lgb_cls = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=3,
                random_state=self.random_state,
                n_estimators=500,
                verbose=-1
            )
            lgb_cls.fit(X_scaled, y_cls)
            
            # 回归模型（回撤）
            lgb_reg_dd = lgb.LGBMRegressor(
                random_state=self.random_state,
                n_estimators=500,
                verbose=-1
            )
            lgb_reg_dd.fit(X_scaled, y_reg_dd)
            
            # 回归模型（上涨）
            lgb_reg_up = lgb.LGBMRegressor(
                random_state=self.random_state,
                n_estimators=500,
                verbose=-1
            )
            lgb_reg_up.fit(X_scaled, y_reg_up)
            
            self.model = {
                'classifier': lgb_cls,
                'regressor_drawdown': lgb_reg_dd,
                'regressor_upswing': lgb_reg_up
            }
            
            results['feature_importance_cls'] = dict(zip(feature_cols, lgb_cls.feature_importances_))
            results['feature_importance_dd'] = dict(zip(feature_cols, lgb_reg_dd.feature_importances_))
            results['feature_importance_up'] = dict(zip(feature_cols, lgb_reg_up.feature_importances_))
        
        return results
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """训练模型"""
        # 移除含NaN的行
        df_clean = df.dropna(subset=['label'])
        
        # 时序划分
        split_idx = int(len(df_clean) * (1 - validation_split))
        train_df = df_clean.iloc[:split_idx]
        val_df = df_clean.iloc[split_idx:]
        
        X_train, feature_cols = self.prepare_features(train_df)
        y_train = self.prepare_labels(train_df)
        
        X_val, _ = self.prepare_features(val_df)
        y_val = self.prepare_labels(val_df)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Features: {len(feature_cols)}")
        
        if self.model_type == 'lightgbm':
            results = self.train_lightgbm_classifier(X_train, y_train, X_val, y_val)
        elif self.model_type == 'catboost':
            results = self.train_catboost_classifier(X_train, y_train, X_val, y_val)
        elif self.model_type == 'multitask':
            results = self.train_multitask_model(df_clean)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # 验证集评估
        val_results = self.evaluate(X_val, y_val, val_df)
        results.update(val_results)
        
        return results
    
    def predict(self, X: np.ndarray) -> Dict:
        """预测"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type == 'multitask':
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            predictions = {}
            predictions['class_probs'] = self.model['classifier'].predict_proba(X)
            predictions['class_pred'] = self.model['classifier'].predict(X)
            predictions['drawdown_pred'] = self.model['regressor_drawdown'].predict(X)
            predictions['upswing_pred'] = self.model['regressor_upswing'].predict(X)
            
            # 转换回原始标签
            predictions['class_pred_original'] = [self.reverse_mapping[int(x)] for x in predictions['class_pred']]
            
        else:
            # LightGBM原生模型预测
            pred_probs = self.model.predict(X)
            pred_classes = np.argmax(pred_probs, axis=1) if pred_probs.ndim > 1 else pred_probs
            
            predictions = {
                'class_probs': pred_probs,
                'class_pred': pred_classes,
                'class_pred_original': [self.reverse_mapping[int(x)] for x in pred_classes]
            }
        
        return predictions
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, df: pd.DataFrame = None) -> Dict:
        """评估模型"""
        predictions = self.predict(X)
        y_pred = predictions['class_pred']
        
        # 转换回原始标签进行评估
        y_true_original = [self.reverse_mapping[x] for x in y_true]
        y_pred_original = predictions['class_pred_original']
        
        results = {}
        
        if HAS_SKLEARN:
            results['classification_report'] = classification_report(y_true_original, y_pred_original, 
                                                             target_names=['Top', 'Neutral', 'Bottom'])
            results['confusion_matrix'] = confusion_matrix(y_true_original, y_pred_original)
        else:
            # 简化评估
            from collections import Counter
            results['prediction_distribution'] = dict(Counter(y_pred_original))
            results['true_distribution'] = dict(Counter(y_true_original))
        
        # 如果是多任务模型，评估回归任务
        if self.model_type == 'multitask' and df is not None:
            dd_true = df['future_drawdown'].fillna(0).values
            up_true = df['future_upswing'].fillna(0).values
            
            if len(dd_true) == len(predictions['drawdown_pred']):
                if HAS_SKLEARN:
                    results['drawdown_mse'] = mean_squared_error(dd_true, predictions['drawdown_pred'])
                    results['upswing_mse'] = mean_squared_error(up_true, predictions['upswing_pred'])
                else:
                    # 简化MSE计算
                    results['drawdown_mse'] = np.mean((dd_true - predictions['drawdown_pred'])**2)
                    results['upswing_mse'] = np.mean((up_true - predictions['upswing_pred'])**2)
        
        return results
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'label_mapping': self.label_mapping,
            'reverse_mapping': self.reverse_mapping
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.scaler = model_data.get('scaler')
        self.label_mapping = model_data['label_mapping']
        self.reverse_mapping = model_data['reverse_mapping']
