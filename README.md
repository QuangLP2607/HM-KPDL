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
git clone https://github.com/QuangLP2607/HM-KPDL
cd HM-KPDL
```

2. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Hướng dẫn chạy

### 1. Xử lý dữ liệu

Chạy các script theo thứ tự để xử lý dữ liệu:

```bash
# Tiền xử lý dữ liệu
python src/data_cleaning/01_preprocess.py

# Kỹ thuật đặc trưng
python src/data_cleaning/02_feature_engineering.py

# Chuẩn hóa phân phối
python src/data_cleaning/03_distribution_normalization.py

# Xử lý giá trị thiếu
python src/data_cleaning/04_missing_value_processing.py

# Xử lý ngoại lai
python src/data_cleaning/05_outlier_processing.py
```

### 2. Huấn luyện mô hình

Chạy các script theo thứ tự để huấn luyện và đánh giá mô hình:

```bash
# Mã hóa đặc trưng
python src/model_training/06_feature_encoding.py

# Chuẩn bị mô hình
python src/model_training/07_model_preparation.py

# Huấn luyện các mô hình
python src/model_training/08_train_models.py

# Đánh giá mô hình
python src/model_training/09_model_evaluation.py
```

### 3. Chạy ứng dụng

Sau khi đã huấn luyện xong các mô hình, chạy ứng dụng web:

```bash
streamlit run src/app.py
```

Ứng dụng sẽ chạy tại địa chỉ: http://localhost:8501

## 📝 Lưu ý

1. Đảm bảo đã cài đặt đầy đủ các thư viện cần thiết từ `requirements.txt`
2. Các file dữ liệu cần được đặt đúng vị trí trong thư mục `src/data/`
3. Các model đã huấn luyện sẽ được lưu trong thư mục `src/models/`
4. Báo cáo đánh giá sẽ được lưu trong thư mục `src/reports/`

## 📁 Cấu trúc dự án

```
HM-KPDL/
├── src/
│   ├── data_cleaning/
│   │   ├── 01_preprocess.py
│   │   ├── 02_feature_engineering.py
│   │   ├── 03_distribution_normalization.py
│   │   ├── 04_missing_value_processing.py
│   │   └── 05_outlier_processing.py
│   ├── model_training/
│   │   ├── 06_feature_encoding.py
│   │   ├── 07_model_preparation.py
│   │   ├── 08_train_models.py
│   │   ├── 09_model_evaluation.py
│   │   └── utils/
│   ├── utils/
│   │   └── load_data.py
│   ├── notebooks/
│   ├── reports/
│   ├── models/
│   ├── data/
│   ├── crawls/
│   └── app.py
├── reports/
├── requirements.txt
└── README.md
```

## 📊 Kết quả

- XGBoost đứng thứ hai với R2 = 0.794
- Random Forest đứng thứ ba với R2 = 0.614, tuy nhiên có hiện tượng overfiting
- Linear Regression chi kết quả thấp với R2 = 0.501

## 👥 Đóng góp
Lương Phúc Quang-20215125

## Đôi lời nhận xét

- Kết quả dự đoán chưa đạt hiệu quả như mong đợi, chủ yếu do mô hình quá tập trung vào khâu tiền xử lý và làm sạch dữ liệu thu thập từ trang Batdongsan.com.vn.

- Trong khi đó, các yếu tố quan trọng như khai thác ngữ nghĩa trong dữ liệu và mở rộng tập dữ liệu từ các nguồn bổ sung (ví dụ: thông tin hạ tầng, kinh tế - xã hội, quy hoạch khu vực) lại chưa được chú trọng đầy đủ.

- Việc thiếu các đặc trưng mang tính bối cảnh đã hạn chế khả năng học và giải thích của mô hình, ảnh hưởng đến chất lượng dự đoán.



