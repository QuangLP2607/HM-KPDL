import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os
from utils.load_data import load_csv

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def encode_categorical_features(df: pd.DataFrame, encoding_config: Dict[str, str]) -> pd.DataFrame:
    """
    Mã hóa các biến phân loại trong DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần mã hóa
    encoding_config : Dict[str, str]
        Dictionary chứa cấu hình mã hóa cho từng cột
        Ví dụ: {'LegalStatus': 'one_hot', 'Furnishing': 'label'}
        
    Returns:
    --------
    pd.DataFrame
        DataFrame đã được mã hóa
    """
    # Danh sách các đặc trưng sẽ sử dụng
    selected_features = [
        'Price', 'Area', 'Bedrooms', 'Bathrooms', 'Floors',
        'AccessWidth', 'FacadeWidth', 'LegalStatus', 'Furnishing',
        'CommuneDensity', 'CommuneCount'
    ]
    
    # Danh sách các biến phân loại cần mã hóa
    categorical_columns = ['LegalStatus', 'Furnishing']
    
    # Kiểm tra và lọc các cột thực sự tồn tại trong DataFrame
    existing_categorical_columns = [col for col in categorical_columns if col in df.columns]
    existing_features = [col for col in selected_features if col in df.columns]
    
    if not existing_categorical_columns:
        logger.warning("Không tìm thấy cột phân loại nào để mã hóa")
        return df[existing_features]
        
    logger.info(f"Các cột sẽ được mã hóa: {existing_categorical_columns}")
    
    # Lọc DataFrame theo các features cần thiết
    df_filtered = df[existing_features].copy()
    
    # Mã hóa từng cột theo cấu hình
    for col in existing_categorical_columns:
        if col not in encoding_config:
            logger.warning(f"Không tìm thấy cấu hình mã hóa cho {col}, sử dụng one_hot encoding")
            encoding_method = 'one_hot'
        else:
            encoding_method = encoding_config[col]
            
        logger.info(f"Mã hóa {col} bằng phương pháp {encoding_method}")
        
        if encoding_method == 'one_hot':
            # One-hot encoding
            dummies = pd.get_dummies(df_filtered[col], prefix=col)
            df_filtered = pd.concat([df_filtered.drop(col, axis=1), dummies], axis=1)
            
        elif encoding_method == 'label':
            # Label encoding
            df_filtered[col] = df_filtered[col].astype('category').cat.codes
            
        elif encoding_method == 'target':
            # Target encoding (mean encoding)
            mean_values = df_filtered.groupby(col)['Price'].mean()
            df_filtered[col] = df_filtered[col].map(mean_values)
            
        elif encoding_method == 'binary':
            # Binary encoding
            dummies = pd.get_dummies(df_filtered[col], prefix=col)
            binary = dummies.apply(lambda x: int(''.join(map(str, x)), 2))
            df_filtered[col] = binary
            
        else:
            logger.warning(f"Phương pháp mã hóa {encoding_method} không được hỗ trợ cho {col}. Sử dụng one_hot encoding.")
            dummies = pd.get_dummies(df_filtered[col], prefix=col)
            df_filtered = pd.concat([df_filtered.drop(col, axis=1), dummies], axis=1)
    
    return df_filtered

def main():
    # Cấu hình
    config = {
        'input': 'data/interim/04_feature_engineering.csv',
        'output_dir': 'data/interim',
        'report_dir': 'reports',
        'encoding_config': {
            'LegalStatus': 'label',  # Mã hóa LegalStatus bằng one-hot encoding
            'Furnishing': 'label'      # Mã hóa Furnishing bằng label encoding
        }
    }
    
    try:
        # Đọc dữ liệu
        logger.info("Đang đọc dữ liệu...")
        df = load_csv(config['input'])
        
        # Mã hóa biến phân loại
        logger.info("Đang mã hóa biến phân loại...")
        df_encoded = encode_categorical_features(df, config['encoding_config'])
        
        # Lưu kết quả
        output_path = os.path.join(config['output_dir'], '05_feature_encoding.csv')
        df_encoded.to_csv(output_path, index=False)
        logger.info(f"Đã lưu kết quả tại: {output_path}")
        
        # Tạo báo cáo
        report_path = os.path.join(config['report_dir'], '05_encoding_analysis.txt')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO MÃ HÓA BIẾN PHÂN LOẠI\n")
            f.write(f"===========================\n\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. Cấu hình mã hóa:\n")
            for col, method in config['encoding_config'].items():
                f.write(f"   - {col}: {method}\n")
            f.write("\n")
            
            f.write("2. Thống kê:\n")
            f.write(f"   - Số lượng biến ban đầu: {len(df.columns)}\n")
            f.write(f"   - Số lượng biến sau khi mã hóa: {len(df_encoded.columns)}\n")
            f.write(f"   - Số lượng biến mới được tạo: {len(df_encoded.columns) - len(df.columns)}\n\n")
            
            f.write("3. Danh sách các biến mới:\n")
            new_columns = set(df_encoded.columns) - set(df.columns)
            for col in sorted(new_columns):
                f.write(f"   - {col}\n")
            
            f.write("\n4. Thông tin về các biến phân loại:\n")
            for col in ['LegalStatus', 'Furnishing']:
                if col in df.columns:
                    f.write(f"\n   {col}:\n")
                    value_counts = df[col].value_counts()
                    for val, count in value_counts.items():
                        f.write(f"      - {val}: {count} bản ghi\n")
        
        logger.info("Hoàn thành mã hóa biến phân loại!")
        
    except Exception as e:
        logger.error(f"Có lỗi xảy ra: {str(e)}")
        raise

if __name__ == "__main__":
    main() 