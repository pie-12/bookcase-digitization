import os
import shutil

img_dir = 'dataset/images/train'
lbl_dir = 'dataset/labels/train'
unlabeled_dir = 'dataset/unlabeled'

if not os.path.exists(unlabeled_dir):
    os.makedirs(unlabeled_dir)

images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
count = 0

for img in images:
    name = os.path.splitext(img)[0]
    label_path = os.path.join(lbl_dir, name + '.txt')
    
    if not os.path.exists(label_path):
        shutil.move(os.path.join(img_dir, img), os.path.join(unlabeled_dir, img))
        count += 1

print(f"Done! Moved {count} unlabeled images to {unlabeled_dir}")
