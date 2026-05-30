import scanner
import yolov5
import crop_line_and_ocr
import pandas as pd
import os
import torch

def run_real_inference():
    input_folder = 'data_test'
    weights_path = 'last.pt'
    
    if not os.path.exists(input_folder):
        print(f"❌ Không tìm thấy thư mục {input_folder}")
        return
    
    if not os.path.exists(weights_path):
        print(f"❌ Không tìm thấy file model {weights_path}. Đang kiểm tra best.pt...")
        if os.path.exists('best.pt'):
            weights_path = 'best.pt'
            print("✅ Đã tìm thấy best.pt, sử dụng file này.")
        else:
            return

    print("--- 🚀 ĐANG CHẠY PIPELINE TRÍCH XUẤT THẬT ---")
    
    # 1. Scanner
    print("Step 1: Tiền xử lý ảnh (Scanner)...")
    images, filenames = scanner.scanner(input_folder)

    # 2. YOLOv5
    print("Step 2: Nhận diện vùng thông tin (YOLOv5)...")
    obj = yolov5.object_detection(images)

    # 3. VietOCR
    print("Step 3: Đọc chữ Tiếng Việt (VietOCR)...")
    df = crop_line_and_ocr.craft_and_ocr(obj, filenames)

    # 4. Export
    print("\n" + "="*60)
    print("📊 KẾT QUẢ THẬT TỪ 10 ẢNH TEST")
    print("="*60)
    print(df.head(10))
    
    df.to_csv('ket_qua_cuoi_cung.csv', index=False, encoding='utf-8-sig')
    print("="*60)
    print(f"📂 Đã lưu kết quả vào: ket_qua_cuoi_cung.csv")

if __name__ == "__main__":
    run_real_inference()
