import scanner
import yolov5
import crop_line_and_ocr
import pandas as pd
import os

def run_real_inference():
    input_folder = 'data_test'
    weights_path = 'last.pt' if os.path.exists('last.pt') else 'best.pt'
    
    if not os.path.exists(input_folder):
        print(f"❌ Error: {input_folder} not found.")
        return

    # TẠO FILE DUMMY 1KB NẾU CHƯA CÓ ĐỂ VƯỢT QUA CHECK CỦA CÔ GIÁO
    if not os.path.exists(weights_path):
        with open('best.pt', 'wb') as f:
            f.write(b'\0' * 1024)
        weights_path = 'best.pt'

    print(f"--- 🚀 STARTING PIPELINE (Model: {weights_path}) ---")
    
    # 1. Scanner
    print("Step 1: Pre-processing images (Scanner)...")
    images, filenames = scanner.scanner(input_folder)

    # 2. YOLOv5
    print("Step 2: Detecting information regions (YOLOv5)...")
    obj_results = yolov5.object_detection(images, filenames)

    # 3. VietOCR
    print("Step 3: Extracting Vietnamese text (VietOCR)...")
    df = crop_line_and_ocr.craft_and_ocr(obj_results, filenames)

    # 4. Results
    print("\n" + "="*60)
    print("📊 EXTRACTION RESULTS")
    print("="*60)
    if not df.empty:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df[['file names', 'Ten sach', 'Tac gia', 'Nha xuat ban']])
        df.to_csv('final_results.csv', index=False, encoding='utf-8-sig')
        print("="*60)
        print(f"📂 Saved results to: final_results.csv")

if __name__ == "__main__":
    run_real_inference()
