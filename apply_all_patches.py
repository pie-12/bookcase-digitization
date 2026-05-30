import os
import shutil

# 1. Patch VietOCR for gdown compatibility
vietocr_utils = r'C:\Users\Admin\AppData\Local\Programs\Python\Python311\Lib\site-packages\vietocr\tool\utils.py'
if os.path.exists(vietocr_utils):
    with open(vietocr_utils, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_line = "return gdown.cached_download(url=url, path=cached, md5=md5, quiet=quiet)"
    new_line = "return gdown.download(url=url, output=cached, quiet=quiet)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(vietocr_utils, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Successfully patched VietOCR locally.")
    else:
        print("VietOCR already patched or line not found.")

# 2. Fix yolov5.py to check for both last.pt and best.pt
yolov5_script = 'yolov5.py'
if os.path.exists(yolov5_script):
    with open(yolov5_script, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if "weights_path = 'last.pt'" in line:
            new_lines.append("    weights_path = 'last.pt' if os.path.exists('last.pt') else 'best.pt'\n")
        else:
            new_lines.append(line)
            
    with open(yolov5_script, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Updated yolov5.py to handle best.pt/last.pt correctly.")

print("All local patches applied.")
