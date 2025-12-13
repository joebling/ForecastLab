"""
ForecastLab - BTC Reversal Event Prediction Model
Core source code package
"""

from .data_source import DataSource
from .feature_engineering import FeatureEngine
from .label_generator import LabelGenerator
from .model_trainer import ReversalModelTrainer
from .model_evaluator import ModelEvaluator
from .predict_reversal import ReversalPredictor

__version__ = "1.0.0"
__author__ = "ForecastLab Team"

__all__ = [
    "DataSource",
    "FeatureEngine", 
    "LabelGenerator",
    "ReversalModelTrainer",
    "ModelEvaluator",
    "ReversalPredictor"
]
