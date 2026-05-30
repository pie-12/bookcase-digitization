import cv2
import torch
import os
import pathlib
import platform
import numpy as np

# Fix PosixPath/WindowsPath conflict globally
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

def object_detection(images, filenames=None):
    # Check for weights file
    weights_path = 'last.pt' if os.path.exists('last.pt') else 'best.pt'
    if not os.path.exists(weights_path):
        print(f"Warning: {weights_path} not found.")
        return []

    # Create output directory for visualization
    output_dir = 'runs/detect'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Colors for different classes (BGR format)
    colors = {
        0: (0, 0, 255),     # Tên sách: Red
        1: (255, 0, 0),     # Tác giả: Blue
        2: (0, 255, 0),     # NXB: Green
        3: (0, 255, 255),   # Tập: Yellow
        4: (255, 0, 255),   # Người dịch: Magenta
        5: (255, 255, 0)    # Tái bản: Cyan
    }
    class_names = {0: 'Ten sach', 1: 'Tac gia', 2: 'NXB', 3: 'Tap', 4: 'Nguoi dich', 5: 'Tai ban'}

    # Load model
    try:
        model = torch.hub.load('yolov5', 'custom', path=weights_path, source='local', force_reload=False)
    except Exception as e:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        
    model.conf = 0.5
    results = []
    
    for i, img in enumerate(images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Create a copy for drawing
        img_draw = img.copy() 
        
        # Inference
        pre = model(img_rgb, size=720)
        locate = pre.pandas().xyxy[0]

        ten_sach, ten_tac_gia, nha_xuat_ban = [], [], []
        tap, nguoi_dich, tai_ban = [], [], []
        lo = []
        
        for index, row in locate.iterrows():
            cls = int(row['class'])
            ymin, ymax = int(row['ymin']), int(row['ymax'])
            xmin, xmax = int(row['xmin']), int(row['xmax'])
            conf = row['confidence']
            
            # Crop image for OCR
            crop = img_rgb[ymin:ymax, xmin:xmax, :]
            
            if cls == 0: ten_sach.append(crop)
            elif cls == 1: ten_tac_gia.append(crop)
            elif cls == 2: nha_xuat_ban.append(crop)
            elif cls == 3: tap.append(crop)
            elif cls == 4: nguoi_dich.append(crop)
            else: tai_ban.append(crop)
            
            lo.append([cls, ymin, ymax, xmin, xmax])
            
            # Draw bounding box and label on img_draw
            color = colors.get(cls, (255, 255, 255))
            label = f"{class_names.get(cls, 'Unknown')} {conf:.2f}"
            cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(img_draw, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save annotated image
        if filenames and i < len(filenames):
            save_path = os.path.join(output_dir, f"detected_{filenames[i]}")
            cv2.imwrite(save_path, img_draw)
        
        # Overlap check
        cache = []
        length = len(lo)
        for j in range(length):
            if lo[j][0] != 0: continue
            idx = lo[j]
            kq = []
            for k in range(length):
                if j == k: continue
                # Check if box k is inside box j
                for y, x in [(1, 3), (1, 4), (2, 3), (2, 4)]:
                    if idx[1] < lo[k][y] < idx[2] and idx[3] < lo[k][x] < idx[4]:
                        kq.append(lo[k][0])
                        break
            cache.extend(list(set(kq)))
        
        features = {0: ten_sach, 1: ten_tac_gia, 2: nha_xuat_ban, 3: tap, 4: nguoi_dich, 5: tai_ban}
        results.append([features, cache])

    return results
