"""
Mining DFS ML Pipeline Package

A machine learning pipeline for predicting stock returns following 
Definitive Feasibility Study (DFS) announcements in the mining sector.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Package-level imports for convenience
from .data_loader import load_data, engineer_features
from .models import train_and_evaluate
from .evaluation import regression_metrics, make_metrics_table

__all__ = [
    'load_data',
    'engineer_features', 
    'train_and_evaluate',
    'regression_metrics',
    'make_metrics_table'
]