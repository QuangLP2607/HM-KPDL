import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.load_data import load_csv

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_and_evaluate_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                            y_train: pd.Series, y_test: pd.Series) -> Dict:
    """
    Huấn luyện và đánh giá các mô hình.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Features cho tập train và test
    y_train, y_test : pd.Series
        Target cho tập train và test
        
    Returns:
    --------
    Dict
        Dictionary chứa các mô hình và kết quả đánh giá
    """
    # Định nghĩa các mô hình
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            objective='reg:squarederror'
        ),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance')
    }
    
    results = {}
    
    # Huấn luyện và đánh giá từng mô hình
    for name, model in models.items():
        logger.info(f"Đang huấn luyện mô hình {name}...")
        
        # Huấn luyện mô hình
        model.fit(X_train, y_train)
        
        # Dự đoán
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Tính toán các metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }
        
        results[name] = {
            'model': model,
            'metrics': metrics
        }
        
        logger.info(f"Hoàn thành huấn luyện mô hình {name}")
        logger.info(f"Test RMSE: {metrics['test_rmse']:,.0f} VND")
        logger.info(f"Test R2: {metrics['test_r2']:.3f}")
    
    return results

def main():
    # Cấu hình
    config = {
        'input_dir': 'data/processed',
        'model_dir': 'models',
        'report_dir': 'reports'
    }
    
    try:
        # Đọc dữ liệu
        logger.info("Đang đọc dữ liệu...")
        X_train = load_csv(os.path.join(config['input_dir'], 'X_train.csv'))
        y_train = load_csv(os.path.join(config['input_dir'], 'y_train.csv'))['Price']
        X_test = load_csv(os.path.join(config['input_dir'], 'X_test.csv'))
        y_test = load_csv(os.path.join(config['input_dir'], 'y_test.csv'))['Price']
        
        # Huấn luyện và đánh giá mô hình
        logger.info("Đang huấn luyện và đánh giá các mô hình...")
        models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Tạo thư mục models nếu chưa tồn tại
        os.makedirs(config['model_dir'], exist_ok=True)
        
        # Lưu tất cả các mô hình
        logger.info("Đang lưu các mô hình...")
        for model_name, model_info in models.items():
            model_path = os.path.join(config['model_dir'], f'{model_name.lower()}_model.joblib')
            joblib.dump(model_info['model'], model_path)
            logger.info(f"Đã lưu mô hình {model_name} tại {model_path}")
        
        # Lưu mô hình tốt nhất
        best_model_name = max(models.items(), key=lambda x: x[1]['metrics']['test_r2'])[0]
        best_model_path = os.path.join(config['model_dir'], 'best_model.joblib')
        joblib.dump(models[best_model_name]['model'], best_model_path)
        logger.info(f"Đã lưu mô hình tốt nhất ({best_model_name}) tại {best_model_path}")
        
        # Tạo báo cáo
        report_path = os.path.join(config['report_dir'], '07_model_evaluation.txt')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO ĐÁNH GIÁ MÔ HÌNH\n")
            f.write(f"=====================\n\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for model_name, model_info in models.items():
                f.write(f"Mô hình: {model_name}\n")
                f.write(f"-------------------\n")
                f.write(f"RMSE: {model_info['metrics']['test_rmse']:.2f}\n")
                f.write(f"R2 Score: {model_info['metrics']['test_r2']:.3f}\n")
                f.write(f"MAE: {model_info['metrics']['test_mae']:.2f}\n")
                f.write(f"MAPE: {model_info['metrics']['test_mae'] / y_test.mean() * 100:.2f}%\n\n")
            
            f.write(f"\nMô hình tốt nhất: {best_model_name}\n")
            f.write(f"R2 Score: {models[best_model_name]['metrics']['test_r2']:.3f}\n")
        
        logger.info("Hoàn thành huấn luyện và đánh giá mô hình!")
        
    except Exception as e:
        logger.error(f"Có lỗi xảy ra: {str(e)}")
        raise

if __name__ == "__main__":
    main() 