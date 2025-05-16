import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils.load_data import load_csv

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_predictions(y_true: pd.Series, y_pred: pd.Series, X: pd.DataFrame) -> Dict:
    """
    Phân tích kết quả dự đoán.
    
    Parameters:
    -----------
    y_true : pd.Series
        Giá thực tế
    y_pred : pd.Series
        Giá dự đoán
    X : pd.DataFrame
        Features
        
    Returns:
    --------
    Dict
        Dictionary chứa các kết quả phân tích
    """
    # Tính toán sai số
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    rel_errors = abs_errors / y_true * 100
    
    # Phân tích theo nhóm giá
    price_bins = pd.qcut(y_true, q=5, labels=['Rất thấp', 'Thấp', 'Trung bình', 'Cao', 'Rất cao'])
    error_by_price = pd.DataFrame({
        'Giá thực tế': y_true,
        'Giá dự đoán': y_pred,
        'Sai số': errors,
        'Sai số tuyệt đối': abs_errors,
        'Sai số tương đối (%)': rel_errors,
        'Nhóm giá': price_bins
    })
    
    # Các trường hợp dự đoán sai nhiều nhất
    worst_predictions = error_by_price.nlargest(10, 'Sai số tương đối (%)')
    
    return {
        'error_by_price': error_by_price,
        'worst_predictions': worst_predictions
    }

def analyze_feature_importance(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Phân tích tầm quan trọng của các biến.
    
    Parameters:
    -----------
    model
        Mô hình đã huấn luyện
    X : pd.DataFrame
        Features
    y : pd.Series
        Target values
        
    Returns:
    --------
    pd.DataFrame
        DataFrame chứa tầm quan trọng của các biến
    """
    if hasattr(model, 'feature_importances_'):
        # Cho Random Forest, XGBoost, LightGBM
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)
    elif hasattr(model, 'coef_'):
        # Cho Linear Regression
        return pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
    else:
        # Cho KNN và các model khác không có feature importance
        # Sử dụng phương pháp permutation importance
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': result.importances_mean
        })
        return importance.sort_values('importance', ascending=False)

def plot_error_distribution(error_by_price: pd.DataFrame, model_name: str, figure_dir: str):
    """Vẽ biểu đồ phân phối sai số."""
    plt.figure(figsize=(10, 6))
    sns.histplot(error_by_price['Sai số'], bins=50)
    plt.title(f'Phân phối sai số dự đoán - {model_name}')
    plt.xlabel('Sai số (VND)')
    plt.savefig(os.path.join(figure_dir, f'error_distribution_{model_name.lower()}.png'))
    plt.close()

def plot_feature_importance(importance: pd.DataFrame, model_name: str, figure_dir: str):
    """Vẽ biểu đồ tầm quan trọng của các biến."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance.head(10), x='importance', y='feature')
    plt.title(f'Top 10 biến quan trọng nhất - {model_name}')
    plt.xlabel('Tầm quan trọng')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'feature_importance_{model_name.lower()}.png'))
    plt.close()

def main():
    # Cấu hình
    config = {
        'input_dir': 'data/processed',
        'model_dir': 'models',
        'report_dir': 'reports',
        'figure_dir': 'reports/figures'
    }
    
    try:
        # Đọc dữ liệu
        logger.info("Đang đọc dữ liệu...")
        X_test = load_csv(os.path.join(config['input_dir'], 'X_test.csv'))
        y_test = load_csv(os.path.join(config['input_dir'], 'y_test.csv'))['Price']
        
        # Tạo thư mục cho hình ảnh
        os.makedirs(config['figure_dir'], exist_ok=True)
        
        # Đọc và đánh giá tất cả các mô hình
        model_files = [f for f in os.listdir(config['model_dir']) if f.endswith('_model.joblib')]
        
        # Tạo báo cáo
        report_path = os.path.join(config['report_dir'], '08_model_analysis.txt')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO PHÂN TÍCH MÔ HÌNH\n")
            f.write(f"=====================\n\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for model_file in model_files:
                model_name = model_file.replace('_model.joblib', '')
                logger.info(f"Đang phân tích mô hình {model_name}...")
                
                # Đọc mô hình
                model = joblib.load(os.path.join(config['model_dir'], model_file))
                
                # Dự đoán
                y_pred = model.predict(X_test)
                
                # Phân tích kết quả
                analysis_results = analyze_predictions(y_test, y_pred, X_test)
                feature_importance = analyze_feature_importance(model, X_test, y_test)
                
                # Vẽ biểu đồ
                plot_error_distribution(analysis_results['error_by_price'], model_name, config['figure_dir'])
                plot_feature_importance(feature_importance, model_name, config['figure_dir'])
                
                # Ghi báo cáo
                f.write(f"\nMô hình: {model_name}\n")
                f.write(f"-------------------\n\n")
                
                f.write("1. Phân tích sai số theo nhóm giá:\n\n")
                error_stats = analysis_results['error_by_price'].groupby('Nhóm giá').agg({
                    'Sai số': ['mean', 'std'],
                    'Sai số tương đối (%)': ['mean', 'std']
                }).round(2)
                f.write(error_stats.to_string())
                f.write("\n\n")
                
                f.write("2. Top 10 trường hợp dự đoán sai nhiều nhất:\n\n")
                f.write(analysis_results['worst_predictions'].to_string())
                f.write("\n\n")
                
                f.write("3. Tầm quan trọng của các biến:\n\n")
                f.write(feature_importance.to_string())
                f.write("\n\n")
                f.write("="*50 + "\n")
        
        logger.info("Hoàn thành phân tích mô hình!")
        
    except Exception as e:
        logger.error(f"Có lỗi xảy ra: {str(e)}")
        raise

if __name__ == "__main__":
    main() 