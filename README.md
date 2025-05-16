# 🏠 PrediHome - Dự đoán giá nhà đất

Ứng dụng dự đoán giá nhà đất sử dụng Machine Learning.

## 📋 Mô tả

PrediHome là một ứng dụng web cho phép người dùng dự đoán giá nhà đất dựa trên các thông số như diện tích, số phòng ngủ, vị trí, và các đặc điểm khác. Ứng dụng sử dụng nhiều mô hình Machine Learning khác nhau để đưa ra dự đoán chính xác nhất.

## 🚀 Tính năng

- Dự đoán giá nhà đất dựa trên nhiều thông số
- Hỗ trợ nhiều mô hình Machine Learning:
  - LightGBM (R2: 0.786)
  - XGBoost (R2: 0.781)
  - Random Forest (R2: 0.769)
  - Linear Regression (R2: 0.518)
  - KNN (R2: 0.612)
- Hiển thị thông tin chi tiết về khu vực
- Giao diện thân thiện với người dùng

## 🛠️ Cài đặt

1. Clone repository:
```bash
git clone https://github.com/.../predihome.git
cd predihome
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Chạy ứng dụng:
```bash
streamlit run src/app.py
```

## 📁 Cấu trúc dự án

```
predihome/
├── src/
│   ├── data_cleaning/
│   │   ├── 01_preprocess.py
│   │   ├── 02_outlier_processing.py
│   │   ├── 03_missing_value_processing.py
│   │   └── 04_feature_engineering.py
│   ├── model_training/
│   │   ├── 05_feature_encoding.py
│   │   ├── 06_prepare_data.py
│   │   ├── 07_train_models.py
│   │   └── 08_model_evaluation.py
│   ├── utils/
│   │   └── load_data.py
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
└── README.md
```

## 🔧 Các bước xử lý dữ liệu

1. **Tiền xử lý dữ liệu** (`01_preprocess.py`)
   - Đọc và kiểm tra dữ liệu
   - Xử lý các giá trị bất thường

2. **Xử lý ngoại lai** (`02_outlier_processing.py`)
   - Phát hiện và xử lý các giá trị ngoại lai
   - Chuẩn hóa dữ liệu

3. **Xử lý giá trị thiếu** (`03_missing_value_processing.py`)
   - Phát hiện và điền giá trị thiếu
   - Xử lý các trường hợp đặc biệt

4. **Kỹ thuật đặc trưng** (`04_feature_engineering.py`)
   - Tạo các đặc trưng mới
   - Chuyển đổi dữ liệu

5. **Mã hóa đặc trưng** (`05_feature_encoding.py`)
   - Mã hóa các biến phân loại
   - Chuẩn hóa các biến số

6. **Chuẩn bị dữ liệu** (`06_prepare_data.py`)
   - Chia tập train/test
   - Lưu trữ dữ liệu đã xử lý
   - Chuẩn bị dữ liệu cho training

7. **Huấn luyện mô hình** (`07_train_models.py`)
   - Huấn luyện nhiều mô hình khác nhau
   - Tối ưu hóa hyperparameters

8. **Đánh giá mô hình** (`08_model_evaluation.py`)
   - Đánh giá hiệu suất mô hình
   - Phân tích kết quả

## 📊 Kết quả

- LightGBM cho kết quả tốt nhất với R2 = 0.786
- XGBoost đứng thứ hai với R2 = 0.781
- Random Forest đứng thứ ba với R2 = 0.769
- Linear Regression và KNN cho kết quả thấp hơn

## 👥 Đóng góp



## 📝 License


