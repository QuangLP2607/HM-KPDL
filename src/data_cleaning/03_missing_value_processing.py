import pandas as pd
import numpy as np
from typing import List, Dict, Union
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from utils.load_data import load_csv

"""
Phương pháp xử lý giá trị thiếu:

1. Giá trị số (Price, Area, Bedrooms, Bathrooms, Floors, AccessWidth, FacadeWidth):
   - Mean/Median: Điền bằng giá trị trung bình/trung vị
   - KNN: Điền dựa trên k giá trị gần nhất
   - Interpolation: Nội suy tuyến tính

2. Giá trị phân loại (Location, PropertyType, LegalStatus, Direction):
   - Mode: Điền bằng giá trị xuất hiện nhiều nhất
   - KNN: Điền dựa trên k giá trị gần nhất
   - Custom: Điền bằng giá trị mặc định

3. Giá trị thời gian (PostDate):
   - Forward/Backward fill: Điền theo thời gian trước/sau
   - Interpolation: Nội suy tuyến tính
"""

def process_missing_values(
    df: pd.DataFrame,
    numeric_columns: List[str] = None,
    categorical_columns: List[str] = None,
    datetime_columns: List[str] = None,
    numeric_method: str = 'mean',
    categorical_method: str = 'mode',
    datetime_method: str = 'ffill',
    plot: bool = False,
    save_path: str = None
) -> pd.DataFrame:
    """
    Xử lý giá trị thiếu trong DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý
    numeric_columns : List[str]
        Danh sách các cột số
    categorical_columns : List[str]
        Danh sách các cột phân loại
    datetime_columns : List[str]
        Danh sách các cột thời gian
    numeric_method : str
        Phương pháp xử lý cho cột số: 'mean', 'median', 'knn', 'interpolate'
    categorical_method : str
        Phương pháp xử lý cho cột phân loại: 'mode', 'knn', 'custom'
    datetime_method : str
        Phương pháp xử lý cho cột thời gian: 'ffill', 'bfill', 'interpolate'
    plot : bool
        Có vẽ đồ thị phân tích không
    save_path : str
        Đường dẫn để lưu kết quả
        
    Returns:
    --------
    pd.DataFrame, Dict
        DataFrame đã được xử lý và dictionary chứa thông tin về giá trị đã điền
    """
    df_clean = df.copy()
    filled_values = {}
    
    # Xử lý giá trị thiếu cho cột số
    if numeric_columns:
        for col in numeric_columns:
            if col not in df_clean.columns:
                print(f"Cột {col} không tồn tại trong DataFrame")
                continue
                
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                print(f"Cột {col} không phải kiểu số")
                continue
                
            missing_count = df_clean[col].isnull().sum()
            if missing_count == 0:
                continue
                
            print(f"\nXử lý giá trị thiếu cho cột {col}:")
            print(f"Số giá trị thiếu: {missing_count}")
            
            if numeric_method == 'mean':
                fill_value = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(fill_value)
                
            elif numeric_method == 'median':
                fill_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(fill_value)
                
            elif numeric_method == 'knn':
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                df_clean[col] = imputer.fit_transform(df_clean[[col]])
                
            elif numeric_method == 'interpolate':
                df_clean[col] = df_clean[col].interpolate(method='linear')
                
            else:
                raise ValueError("Phương pháp không hợp lệ cho cột số")
                
            filled_values[col] = {
                'method': numeric_method,
                'missing_count': missing_count,
                'fill_value': fill_value if numeric_method in ['mean', 'median'] else None
            }
            
            if plot:
                plt.figure(figsize=(12, 4))
                
                # Histogram trước và sau khi điền
                plt.subplot(1, 2, 1)
                sns.histplot(df[col].dropna(), bins=50, kde=True, label='Trước khi điền')
                sns.histplot(df_clean[col], bins=50, kde=True, label='Sau khi điền')
                plt.title(f'Phân bố của {col}')
                plt.legend()
                
                # Boxplot trước và sau khi điền
                plt.subplot(1, 2, 2)
                sns.boxplot(data=pd.DataFrame({
                    'Trước khi điền': df[col].dropna(),
                    'Sau khi điền': df_clean[col]
                }))
                plt.title(f'Boxplot của {col}')
                
                plt.tight_layout()
                plt.show()
    
    # Xử lý giá trị thiếu cho cột phân loại
    if categorical_columns:
        for col in categorical_columns:
            if col not in df_clean.columns:
                print(f"Cột {col} không tồn tại trong DataFrame")
                continue
                
            missing_count = df_clean[col].isnull().sum()
            if missing_count == 0:
                continue
                
            print(f"\nXử lý giá trị thiếu cho cột {col}:")
            print(f"Số giá trị thiếu: {missing_count}")
            
            if categorical_method == 'mode':
                fill_value = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(fill_value)
                
            elif categorical_method == 'knn':
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                df_clean[col] = imputer.fit_transform(df_clean[[col]])
                
            elif categorical_method == 'custom':
                # Điền giá trị mặc định tùy theo cột
                default_values = {
                    'Location': 'Unknown',
                    'PropertyType': 'Unknown',
                    'LegalStatus': 'Unknown',
                    'Direction': 'Unknown'
                }
                fill_value = default_values.get(col, 'Unknown')
                df_clean[col] = df_clean[col].fillna(fill_value)
                
            else:
                raise ValueError("Phương pháp không hợp lệ cho cột phân loại")
                
            filled_values[col] = {
                'method': categorical_method,
                'missing_count': missing_count,
                'fill_value': fill_value if categorical_method in ['mode', 'custom'] else None
            }
            
            if plot:
                plt.figure(figsize=(12, 4))
                
                # Bar plot trước và sau khi điền
                plt.subplot(1, 2, 1)
                df[col].value_counts().plot(kind='bar', label='Trước khi điền')
                plt.title(f'Phân bố của {col} (Trước)')
                plt.xticks(rotation=45)
                
                plt.subplot(1, 2, 2)
                df_clean[col].value_counts().plot(kind='bar', label='Sau khi điền')
                plt.title(f'Phân bố của {col} (Sau)')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.show()
    
    # Xử lý giá trị thiếu cho cột thời gian
    if datetime_columns:
        for col in datetime_columns:
            if col not in df_clean.columns:
                print(f"Cột {col} không tồn tại trong DataFrame")
                continue
                
            missing_count = df_clean[col].isnull().sum()
            if missing_count == 0:
                continue
                
            print(f"\nXử lý giá trị thiếu cho cột {col}:")
            print(f"Số giá trị thiếu: {missing_count}")
            
            if datetime_method == 'ffill':
                df_clean[col] = df_clean[col].fillna(method='ffill')
                
            elif datetime_method == 'bfill':
                df_clean[col] = df_clean[col].fillna(method='bfill')
                
            elif datetime_method == 'interpolate':
                df_clean[col] = df_clean[col].interpolate(method='time')
                
            else:
                raise ValueError("Phương pháp không hợp lệ cho cột thời gian")
                
            filled_values[col] = {
                'method': datetime_method,
                'missing_count': missing_count
            }
    
    # Lưu kết quả nếu có đường dẫn
    if save_path:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Lưu DataFrame
        df_clean.to_csv(save_path, index=False)
    
    return df_clean, filled_values

def main():
    # Cấu hình
    config = {
        'numeric_method': 'mean',    # 'mean', 'median', 'knn', 'interpolate'
        'categorical_method': 'mode', # 'mode', 'knn', 'custom'
        'datetime_method': 'ffill',   # 'ffill', 'bfill', 'interpolate'
        'plot': True,                # True/False
        'save': True,                # True/False
        'input': 'data/interim/02_processed_iqr.csv',
        'output_dir': 'data/interim',
        'report_dir': 'reports'
    }
    
    # Đọc dữ liệu sử dụng hàm load_csv
    df = load_csv(config['input'])
    
    # Định nghĩa các cột theo loại
    numeric_columns = ['Price', 'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'AccessWidth', 'FacadeWidth']
    categorical_columns = ['Location', 'PropertyType', 'LegalStatus', 'Direction']
    datetime_columns = ['PostDate']
    
    # Tạo đường dẫn lưu file
    save_path = None
    report_path = None
    if config['save']:
        # Lưu file CSV vào thư mục data
        save_path = os.path.join(config['output_dir'], f'03_processed_missing.csv')
        # Lưu file báo cáo vào thư mục reports
        report_path = os.path.join(config['report_dir'], f'03_missing_value_analysis.txt')
    
    # Xử lý giá trị thiếu
    df_clean, filled_values = process_missing_values(
        df,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        numeric_method=config['numeric_method'],
        categorical_method=config['categorical_method'],
        datetime_method=config['datetime_method'],
        plot=config['plot'],
        save_path=save_path
    )
    
    # Lưu báo cáo nếu có đường dẫn
    if report_path:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO XỬ LÝ GIÁ TRỊ THIẾU\n")
            f.write(f"==========================\n\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. Cấu hình xử lý:\n")
            f.write(f"   - Phương pháp cho cột số: {config['numeric_method']}\n")
            f.write(f"   - Phương pháp cho cột phân loại: {config['categorical_method']}\n")
            f.write(f"   - Phương pháp cho cột thời gian: {config['datetime_method']}\n\n")
            
            f.write("2. Thống kê giá trị thiếu:\n")
            for col, info in filled_values.items():
                f.write(f"\n   {col}:\n")
                f.write(f"   - Số giá trị thiếu: {info['missing_count']}\n")
                f.write(f"   - Phương pháp xử lý: {info['method']}\n")
                if 'fill_value' in info and info['fill_value'] is not None:
                    f.write(f"   - Giá trị điền: {info['fill_value']}\n")
    
    print("\nĐã xử lý giá trị thiếu xong!")
    if save_path:
        print(f"\nDữ liệu đã được lưu tại: {save_path}")
    if report_path:
        print(f"Báo cáo đã được lưu tại: {report_path}")

if __name__ == "__main__":
    main() 