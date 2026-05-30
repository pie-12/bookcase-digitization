import cv2
import torch
import os
import pathlib
import platform

# Fix PosixPath/WindowsPath conflict globally
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

def object_detection(images):
    # Check for weights file
    weights_path = 'last.pt' if os.path.exists('last.pt') else 'best.pt'
    if not os.path.exists(weights_path):
        print(f"Warning: {weights_path} not found.")
        return []

    # Load model
    # Note: yolov5 folder must be in root or in PYTHONPATH
    try:
        model = torch.hub.load('yolov5', 'custom', path=weights_path, source='local', force_reload=False)
    except Exception as e:
        print(f"Error loading model from local: {e}")
        print("Attempting to load using ultralytics repo...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        
    model.conf = 0.5
    results = []
    
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Inference
        pre = model(img_rgb, size=720)
        locate = pre.pandas().xyxy[0]

        ten_sach = []
        ten_tac_gia = []
        nha_xuat_ban = []
        tap = []
        nguoi_dich = []
        tai_ban = []
        lo = []
        
        for index, row in locate.iterrows():
            cls = int(row['class'])
            crop = img_rgb[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax']), :]
            
            if cls == 0: ten_sach.append(crop)
            elif cls == 1: ten_tac_gia.append(crop)
            elif cls == 2: nha_xuat_ban.append(crop)
            elif cls == 3: tap.append(crop)
            elif cls == 4: nguoi_dich.append(crop)
            else: tai_ban.append(crop)
            
            lo.append([cls, int(row['ymin']), int(row['ymax']), int(row['xmin']), int(row['xmax'])])
        
        # Overlap check
        cache = []
        length = len(lo)
        for i in range(length):
            if lo[i][0] != 0: continue
            idx = lo[i]
            kq = []
            for j in range(length):
                if i == j: continue
                # Check if box j is inside box i
                for y, x in [(1, 3), (1, 4), (2, 3), (2, 4)]:
                    if idx[1] < lo[j][y] < idx[2] and idx[3] < lo[j][x] < idx[4]:
                        kq.append(lo[j][0])
                        break
            cache.extend(list(set(kq)))
        
        features = {0: ten_sach, 1: ten_tac_gia, 2: nha_xuat_ban, 3: tap, 4: nguoi_dich, 5: tai_ban}
        results.append([features, cache])

    return results
