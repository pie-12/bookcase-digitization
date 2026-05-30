import os

file_path = r'C:\Users\Admin\AppData\Local\Programs\Python\Python311\Lib\site-packages\craft_text_detector\models\basenet\vgg16_bn.py'

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_content = []
for line in lines:
    if "from torchvision.models.vgg import model_urls" in line:
        new_content.append("try:\n")
        new_content.append("    from torchvision.models.vgg import model_urls\n")
        new_content.append("except ImportError:\n")
        new_content.append("    model_urls = {'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'}\n")
    else:
        new_content.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_content)

print("Successfully patched vgg16_bn.py locally.")
