# 🚀 Hướng dẫn chạy Dự án Bookcase Digitization trên Google Colab

Dựa trên tài liệu kỹ thuật của nhóm, đây là quy trình chuẩn để thiết lập và chạy toàn bộ Pipeline.

### 1. Cài đặt môi trường (Cell 1)
```python
# 1. Clone dự án và cài đặt thư viện
!git clone https://github.com/pie-12/bookcase-digitization.git
%cd bookcase-digitization

# 2. Cài đặt các thư viện lõi
!pip install -q craft-text-detector vietocr==0.3.5
!pip install -r requirements.txt

# 3. Clone repo YOLOv5 để lấy script train/detect
!git clone https://github.com/ultralytics/yolov5
```

### 2. Huấn luyện lại YOLOv5x6 (Cell 2)
Theo tài liệu `@YOLO-object-detection.pdf`, nhóm sử dụng model **YOLOv5x6**. Nếu file `last.pt` bị thiếu, hãy chạy lệnh này sau khi đã chuẩn bị dữ liệu mới:
```python
%cd /content/bookcase-digitization/yolov5
# Huấn luyện nhanh với bộ dữ liệu mới (đã gán nhãn trong thư mục data)
!python train.py --img 720 --batch 8 --epochs 50 --data ../models/data.yaml --weights yolov5x6.pt

# Sau khi train xong, copy file weights ra ngoài
!cp runs/train/exp/weights/best.pt ../last.pt
%cd ..
```

### 3. Cấu hình VietOCR (TransformerOCR)
Mã nguồn đã được cập nhật để sử dụng `vgg_transformer` mặc định. Theo tài liệu `@vietocr.pdf`, bạn có thể tùy chỉnh cấu hình như sau:
```python
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained'] = True
config['device'] = 'cuda:0'
detector = Predictor(config)
```

### 4. Chạy toàn bộ Pipeline (Cell 3)
```python
# Chạy nhận diện và OCR cho toàn bộ ảnh trong thư mục data_test
!python main.py -i data_test

# Hiển thị kết quả trích xuất
import pandas as pd
df = pd.read_csv('data.csv')
display(df)
```

---
**Lưu ý:** Nếu không kịp train YOLO, hãy sử dụng model pretrained `yolov5s.pt` để demo luồng xử lý trước khi có kết quả từ model `x6` nặng hơn.
