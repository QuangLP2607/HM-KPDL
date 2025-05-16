import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from utils.load_data import load_csv

# Cấu hình trang
st.set_page_config(
    page_title="Dự đoán giá nhà",
    page_icon="🏠",
    layout="wide"
)

# Tiêu đề
st.title("🏠 Dự đoán giá nhà đất")
st.markdown("---")

# Hàm chuyển đổi giá trị LegalStatus
def convert_legal_status_to_label(status: str) -> int:
    """
    Chuyển đổi giá trị LegalStatus từ text sang số theo label encoding
    """
    status_map = {
        "Có": 1,
        "Không": 0
    }
    return status_map.get(status, 0)  # Mặc định là "Không" nếu không tìm thấy

# Hàm chuyển đổi giá trị Furnishing
def convert_furnishing_to_label(furnishing: str) -> int:
    """
    Chuyển đổi giá trị Furnishing từ text sang số theo label encoding
    """
    furnishing_map = {
        "Có": 1,
        "Không": 0
    }
    return furnishing_map.get(furnishing, 0)  # Mặc định là "Không" nếu không tìm thấy

# Load dữ liệu địa lý
@st.cache_resource
def load_location_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wiki_path = os.path.join(base_dir, 'data', 'raw', 'wiki.json')
    if not os.path.exists(wiki_path):
        st.error(f"Không tìm thấy file wiki.json tại: {wiki_path}")
        return {}
    with open(wiki_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load model và scaler
@st.cache_resource
def load_models_and_scalers():
    # Đường dẫn tuyệt đối đến thư mục models
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(models_dir):
        st.error(f"Không tìm thấy thư mục models tại: {models_dir}")
        return {}, {}
    
    # Dictionary để lưu các model
    models = {}
    
    # Tìm tất cả các file model trong thư mục
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') and f != 'scalers.joblib']
    
    # Thông tin về các model
    model_info = {
        'lightgbm_model.joblib': 'LightGBM (Best - R2: 0.786)',
        'xgboost_model.joblib': 'XGBoost (R2: 0.781)',
        'random forest_model.joblib': 'Random Forest (R2: 0.769)',
        'linear regression_model.joblib': 'Linear Regression (R2: 0.518)',
        'knn_model.joblib': 'KNN (R2: 0.612)',
        'best_model.joblib': 'Best Model (R2: 0.786)'
    }
    
    # Load tất cả các model
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        try:
            model = joblib.load(model_path)
            # Lấy tên hiển thị từ model_info hoặc sử dụng tên file
            display_name = model_info.get(model_file, model_file.replace('_model.joblib', '').title())
            models[display_name] = model
            # st.success(f"Đã tải thành công model: {display_name}")
        except Exception as e:
            st.warning(f"Không thể tải model {model_file}: {str(e)}")
    
    # Load scalers
    scaler_path = os.path.join(models_dir, 'scalers.joblib')
    if not os.path.exists(scaler_path):
        st.error(f"Không tìm thấy file scaler tại: {scaler_path}")
        return models, {}
    
    try:
        scalers = joblib.load(scaler_path)
        return models, scalers
    except Exception as e:
        st.error(f"Lỗi khi tải scaler: {str(e)}")
        return models, {}

try:
    models, scalers = load_models_and_scalers()
    location_data = load_location_data()
    if not models:
        st.error("Không tìm thấy model nào để dự đoán.")
        st.stop()
except Exception as e:
    st.error("Không thể tải mô hình hoặc dữ liệu địa lý. Vui lòng kiểm tra lại đường dẫn và file.")
    st.stop()

# Chọn model
st.subheader("Chọn mô hình dự đoán")
model_names = list(models.keys())
selected_model = st.selectbox(
    "Mô hình",
    model_names,
    index=0,
    help="Chọn mô hình để dự đoán giá nhà"
)

# Hiển thị thông tin về model được chọn
if selected_model:
    st.info(f"""
    **Thông tin mô hình:**
    - {selected_model}
    - Độ chính xác (R2) được tính trên tập test
    """)

# Chọn địa chỉ (tách riêng khỏi form)
st.subheader("Địa chỉ")
if not location_data:
    st.error("Không có dữ liệu địa lý. Vui lòng kiểm tra lại file wiki.json")
    st.stop()

provinces = list(location_data.keys())
selected_province = st.selectbox(
    "Tỉnh/Thành phố",
    provinces,
    help="Chọn tỉnh/thành phố để xem danh sách quận/huyện và thông tin chi tiết"
)

# Cập nhật danh sách quận/huyện dựa trên tỉnh/thành phố được chọn
districts = list(location_data[selected_province].keys())
selected_district = st.selectbox(
    "Quận/Huyện",
    districts,
    help="Thông tin sẽ được cập nhật tự động khi chọn quận/huyện"
)

# Lấy thông tin về quận/huyện được chọn
district_info = location_data[selected_province][selected_district]
commune_density = float(district_info['distribute'].replace('.', ''))
commune_count = int(district_info['communes'])

# Hiển thị thông tin về quận/huyện
st.info(f"""
**Thông tin {selected_district.title()}:**
- Mật độ dân số: {commune_density:,.0f} người/km²
- Số phường/xã: {commune_count}
- Diện tích: {district_info['area']} km²
- Dân số: {district_info['number_people']} người

*Thông tin sẽ tự động cập nhật khi bạn thay đổi tỉnh/thành phố hoặc quận/huyện*
""")

# Tạo form nhập liệu
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Thông tin cơ bản")
        area = st.number_input("Diện tích (m²)", min_value=0.0, value=100.0)
        bedrooms = st.number_input("Số phòng ngủ", min_value=0, value=2)
        bathrooms = st.number_input("Số phòng tắm", min_value=0, value=1)
        floors = st.number_input("Số tầng", min_value=0, value=1)
        
    with col2:
        st.subheader("Thông tin bổ sung")
        access_width = st.number_input("Đường vào (m)", min_value=0.0, value=4.0)
        facade_width = st.number_input("Mặt tiền (m)", min_value=0.0, value=5.0)
        
    st.subheader("Trạng thái")
    col3, col4 = st.columns(2)
    
    with col3:
        legal_status = st.radio("Pháp lý", ["Có", "Không"])
        
    with col4:
        furnishing = st.radio("Nội thất", ["Có", "Không"])
    
    # Nút dự đoán
    submit_button = st.form_submit_button("Dự đoán giá")

# Xử lý dự đoán
if submit_button:
    try:
        # Tạo DataFrame từ input với thứ tự cột cố định
        input_data = pd.DataFrame({
            'Area': [area],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Floors': [floors],
            'AccessWidth': [access_width],
            'FacadeWidth': [facade_width],
            'LegalStatus': [convert_legal_status_to_label(legal_status)],
            'Furnishing': [convert_furnishing_to_label(furnishing)],
            'CommuneDensity': [commune_density],
            'CommuneCount': [commune_count]   
        })
        
        # Chuẩn hóa dữ liệu
        numeric_cols = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'AccessWidth', 
                       'FacadeWidth', 'CommuneDensity', 'CommuneCount']
        
        # Kiểm tra và áp dụng scaler cho từng cột
        for col in numeric_cols:
            if col in scalers:
                try:
                    input_data[col] = scalers[col].transform(input_data[[col]])
                except Exception as e:
                    st.warning(f"Không thể chuẩn hóa cột {col}: {str(e)}")
        
        # Dự đoán với model đã chọn
        model = models[selected_model]
        
        # Đảm bảo thứ tự cột giống với training data
        expected_columns = [
            'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'AccessWidth', 
            'FacadeWidth',  'LegalStatus', 'Furnishing','CommuneDensity', 'CommuneCount'
           
        ]
        input_data = input_data[expected_columns]
        
        # In ra thông tin debug
        # st.write("Debug - Input data columns:", input_data.columns.tolist())
        # st.write("Debug - Input data shape:", input_data.shape)
        
        # Dự đoán
        prediction = model.predict(input_data)[0]
        
        # Hiển thị kết quả
        st.markdown("---")
        st.subheader("Kết quả dự đoán")
        
        # Tạo 3 cột để hiển thị kết quả
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Giá dự đoán", f"{prediction:,.2f} tỷ đồng")
        
        with col2:
            st.metric("Giá/m²", f"{(prediction * 1000 / area):,.0f} triệu đồng/m²")
        
        with col3:
            st.metric("Thời gian dự đoán", datetime.now().strftime("%H:%M:%S"))
        
        # Hiển thị thông tin chi tiết
        st.markdown("### Thông tin chi tiết")
        st.markdown(f"""
        - Mô hình sử dụng: {selected_model}
        - Địa chỉ: {selected_district.title()}, {selected_province.title()}
        - Diện tích: {area:,.1f} m²
        - Số phòng ngủ: {bedrooms}
        - Số phòng tắm: {bathrooms}
        - Số tầng: {floors}
        - Đường vào: {access_width:,.1f} m
        - Mặt tiền: {facade_width:,.1f} m
        - Mật độ dân số: {commune_density:,.0f} người/km²
        - Số phường/xã: {commune_count}
        - Pháp lý: {legal_status}
        - Nội thất: {furnishing}
        """)
        
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi dự đoán: {str(e)}")
        st.error("Chi tiết lỗi:")
        st.error(str(e.__class__.__name__))
        st.error(str(e))
        # In thêm thông tin debug
        st.write("Debug - Input data:", input_data)
        st.write("Debug - Model type:", type(model))
        st.write("Debug - Model features:", getattr(model, 'feature_names_', None))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Được phát triển bởi HM&KPDL Team</p>
    <p>© 2024 - Phiên bản 1.0</p>
</div>
""", unsafe_allow_html=True) 