import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.load_data import load_csv

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_data_for_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Chuẩn bị dữ liệu cho việc huấn luyện mô hình.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần chuẩn bị
    test_size : float
        Tỷ lệ dữ liệu test
    random_state : int
        Random seed để đảm bảo tính tái lập
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]
        X_train, X_test, y_train, y_test và dictionary chứa các scaler
    """
    # Tách features và target
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Chia giá thành các nhóm
    price_bins = pd.qcut(y, q=10, labels=False)  # Chia thành 10 nhóm có số lượng bằng nhau
    
    # Danh sách các cột số cần chuẩn hóa
    numeric_columns = [
        'Area', 'Bedrooms', 'Bathrooms', 'Floors',
        'AccessWidth', 'FacadeWidth', 'CommuneDensity', 'CommuneCount'
    ]
    
    # Chuẩn hóa các cột số
    scalers = {}
    for col in numeric_columns:
        if col in X.columns:
            scaler = StandardScaler()
            X[col] = scaler.fit_transform(X[[col]])
            scalers[col] = scaler
    
    # Chia dữ liệu thành train và test dựa trên nhóm giá
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=price_bins
    )
    
    return X_train, X_test, y_train, y_test, scalers

def main():
    # Cấu hình
    config = {
        'input': 'data/interim/05_feature_encoding.csv',
        'output_dir': 'data/processed',
        'model_dir': 'models',
        'report_dir': 'reports'
    }
    
    try:
        # Đọc dữ liệu
        logger.info("Đang đọc dữ liệu...")
        df = load_csv(config['input'])
        
        # Chuẩn bị dữ liệu
        logger.info("Đang chuẩn bị dữ liệu...")
        X_train, X_test, y_train, y_test, scalers = prepare_data_for_model(df)
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(config['model_dir'], exist_ok=True)
        
        # Lưu dữ liệu đã chuẩn bị
        X_train.to_csv(os.path.join(config['output_dir'], 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(config['output_dir'], 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(config['output_dir'], 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(config['output_dir'], 'y_test.csv'), index=False)
        
        # Lưu scalers
        joblib.dump(scalers, os.path.join(config['model_dir'], 'scalers.joblib'))
        
        # Tạo báo cáo
        report_path = os.path.join(config['report_dir'], '06_data_preparation.txt')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO CHUẨN BỊ DỮ LIỆU\n")
            f.write(f"========================\n\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. Thống kê dữ liệu:\n")
            f.write(f"   - Tổng số mẫu: {len(df)}\n")
            f.write(f"   - Số mẫu train: {len(X_train)}\n")
            f.write(f"   - Số mẫu test: {len(X_test)}\n\n")
            
            f.write("2. Thông tin về các biến:\n")
            f.write(f"   - Số lượng features: {X_train.shape[1]}\n")
            f.write(f"   - Danh sách features:\n")
            for col in X_train.columns:
                f.write(f"      - {col}\n")
            
            f.write("\n3. Thống kê giá nhà:\n")
            f.write(f"   - Giá trung bình (train): {y_train.mean():,.0f} VND\n")
            f.write(f"   - Giá thấp nhất (train): {y_train.min():,.0f} VND\n")
            f.write(f"   - Giá cao nhất (train): {y_train.max():,.0f} VND\n")
            f.write(f"   - Giá trung bình (test): {y_test.mean():,.0f} VND\n")
            f.write(f"   - Giá thấp nhất (test): {y_test.min():,.0f} VND\n")
            f.write(f"   - Giá cao nhất (test): {y_test.max():,.0f} VND\n")
        
        logger.info("Hoàn thành chuẩn bị dữ liệu!")
        
    except Exception as e:
        logger.error(f"Có lỗi xảy ra: {str(e)}")
        raise

if __name__ == "__main__":
    main() 