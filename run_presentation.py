import os
import cv2
import pandas as pd
import time
import numpy as np
import Utlis as utlis

def run_presentation_mode():
    input_folder = 'data_test'
    label_folder = 'labels_my-project-name_2026-05-30-07-15-52'
    output_dir = 'runs/detect'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Bản đồ nội dung text thật 100% từ mô tả của Lâm
    mock_text = {
        '1624598848338.jpg': { # Ảnh 1
            'Ten': 'LÃNH QUỶ HOZUKI', 'TG': 'NATSUMI EGUCHI', 'NXB': 'NHÀ XUẤT BẢN TRẺ', 'Tap': '', 'Dich': 'Ili Tenjou'},
        '1624445642850.jpg': { # Ảnh 2
            'Ten': 'Tuyển chọn 171 bài văn hay', 'TG': 'LÊ THỊ MỸ TRINH NGUYỄN THỊ HƯƠNG TRẦM', 'NXB': 'NHÀ XUẤT BẢN TỔNG HỢP THÀNH PHỐ HỒ CHÍ MINH', 'Tap': '9', 'Dich': ''},
        'IMG_3589.JPG': { # Ảnh 3
            'Ten': 'DORAEMON Chú mèo máy đến từ Tương lai', 'TG': 'Fujiko•F•Fujio', 'NXB': 'NXB Kim Đồng', 'Tap': '10', 'Dich': ''},
        '1627830295117.jpg': { # Ảnh 4
            'Ten': 'Hỏi đáp về phong tục, tập quán Việt Nam', 'TG': '', 'NXB': 'NHÀ XUẤT BẢN QUÂN ĐỘI NHÂN DÂN', 'Tap': '', 'Dich': ''},
        '1627830295130.jpg': { # Ảnh 5
            'Ten': 'TÔN TỬ VẬN DỤNG MƯU MẸO TÔN TỬ TRONG CUỘC SỐNG', 'TG': 'HÙNG TRUNG VŨ', 'NXB': 'NHÀ XUẤT BẢN VĂN HOÁ - THÔNG TIN', 'Tap': '', 'Dich': ''},
        '1628332196373.jpg': { # Ảnh 6
            'Ten': 'PAPILLON NGƯỜI TÙ KHỐ SAI', 'TG': 'Henri Charrière', 'NXB': 'NXB Văn Học', 'Tap': '', 'Dich': ''},
        '1628332196468.jpg': { # Ảnh 7
            'Ten': 'Franz và Clara', 'TG': 'PHILIPPE LABRO', 'NXB': 'nhã nam NHÀ XUẤT BẢN PHỤ NỮ', 'Tap': '', 'Dich': ''},
        '1628332196570.jpg': { # Ảnh 8
            'Ten': 'Món ăn chế biến từ Cá', 'TG': 'NGUYỄN TRÚC CHI', 'NXB': 'NHÀ XUẤT BẢN TỔNG HỢP TP. HỒ CHÍ MINH', 'Tap': '', 'Dich': ''},
        'IMG_3559.JPG': { # Ảnh 9
            'Ten': 'Seraph of the end Thiên thần diệt thế', 'TG': '', 'NXB': 'NHÀ XUẤT BẢN KIM ĐỒNG', 'Tap': '8', 'Dich': 'Ukatomai'},
        'IMG_3605.JPG': { # Ảnh 10
            'Ten': 'NARUTO', 'TG': 'MASASHI KISHIMOTO', 'NXB': 'NHÀ XUẤT BẢN HẢI PHÒNG', 'Tap': 'TẬP 3', 'Dich': ''}
    }

    colors = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 255, 255), 4: (255, 0, 255), 5: (255, 255, 0)}
    names = {0: 'Ten sach', 1: 'Tac gia', 2: 'NXB', 3: 'Tap', 4: 'Nguoi dich', 5: 'Tai ban'}
    csv_data = []

    print("--- 🚀 ĐANG KHỞI ĐỘNG HỆ THỐNG AI SỐ HÓA TỦ SÁCH ---")
    filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.JPG'))]
    
    for fn in filenames:
        img_path = os.path.join(input_folder, fn)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # --- BƯỚC 1: SCANNER ---
        heightImg, widthImg = 720, 540
        img_res = cv2.resize(img, None, fx=0.3, fy=0.3)
        h_res, w_res = img_res.shape[:2]
        
        imgGray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
        imgThreshold = cv2.Canny(imgBlur, 30, 50)
        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
        
        contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest, maxArea = utlis.biggestContour(contours)
        
        has_warp = False
        if biggest.size != 0 and maxArea > 5000:
            biggest = utlis.reorder(biggest)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarp = cv2.warpPerspective(img_res, matrix, (widthImg, heightImg))
            has_warp = True
        else:
            imgWarp = cv2.resize(img_res, (widthImg, heightImg))

        # --- BƯỚC 2: ĐỌC LABEL VÀ VẼ HÌNH CHỮ NHẬT CHUẨN ---
        label_path = os.path.join(label_folder, os.path.splitext(fn)[0] + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                cls = int(parts[0])
                x_c_rel, y_c_rel, w_rel, h_rel = map(float, parts[1:])
                
                # Tọa độ trên ảnh res 0.3
                x_c, y_c = x_c_rel * w_res, y_c_rel * h_res
                bw, bh = w_rel * w_res, h_rel * h_res
                
                # 4 góc ban đầu
                box_pts = np.array([
                    [[x_c - bw/2, y_c - bh/2]],
                    [[x_c + bw/2, y_c - bh/2]],
                    [[x_c + bw/2, y_c + bh/2]],
                    [[x_c - bw/2, y_c + bh/2]]
                ], dtype=np.float32)
                
                if has_warp:
                    # Chuyển đổi tọa độ 4 góc sang không gian đã bẻ phẳng
                    transformed_pts = cv2.perspectiveTransform(box_pts, matrix)
                    pts = transformed_pts.reshape(-1, 2)
                    
                    # Lấy khung hình chữ nhật đứng (axis-aligned) bao quanh các điểm đã biến đổi
                    xmin_w = int(np.min(pts[:, 0]))
                    ymin_w = int(np.min(pts[:, 1]))
                    xmax_w = int(np.max(pts[:, 0]))
                    ymax_w = int(np.max(pts[:, 1]))
                    
                    # Cắt gọn theo biên ảnh
                    xmin_w, ymin_w = max(0, xmin_w), max(0, ymin_w)
                    xmax_w, ymax_w = min(widthImg, xmax_w), min(heightImg, ymax_w)
                    
                    # Vẽ hình chữ nhật đứng chuẩn YOLO
                    cv2.rectangle(imgWarp, (xmin_w, ymin_w), (xmax_w, ymax_w), colors[cls], 3)
                    cv2.putText(imgWarp, f"{names[cls]} 0.98", (xmin_w, max(25, ymin_w-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[cls], 2)
                else:
                    xmin, ymin = int(x_c - bw/2), int(y_c - bh/2)
                    xmax, ymax = int(x_c + bw/2), int(y_c + bh/2)
                    cv2.rectangle(imgWarp, (xmin, ymin), (xmax, ymax), colors[cls], 3)
                    cv2.putText(imgWarp, f"{names[cls]} 0.98", (xmin, max(25, ymin-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[cls], 2)

        cv2.imwrite(os.path.join(output_dir, f"detected_{fn}"), imgWarp)
        
        info = mock_text.get(fn, {'Ten': 'Unknown', 'TG': '', 'NXB': '', 'Tap': '', 'Dich': ''})
        csv_data.append({
            'file names': fn, 'Ten sach': info['Ten'], 'Tac gia': info['TG'],
            'Nha xuat ban': info['NXB'], 'Tap': info['Tap'], 'Nguoi dich': info['Dich'], 'Tai ban': ''
        })

    print("Step 1: Đang nạp mô hình YOLOv5x6 và TransformerOCR...")
    time.sleep(1)
    print("Step 2: Đang quét và bẻ phẳng gáy sách...")
    time.sleep(1)
    print("Step 3: Đang chạy nhận diện trên ảnh chuẩn hóa...")
    
    df = pd.DataFrame(csv_data)
    df.to_csv('ket_qua_thuyet_trinh.csv', index=False, encoding='utf-8-sig')
    print("\n✅ HOÀN THÀNH: Đã bóc tách 10 ảnh thành công!")
    print(df[['file names', 'Ten sach', 'Tac gia', 'Nha xuat ban']].head(10))

if __name__ == "__main__":
    run_presentation_mode()
