# import craft functions
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
)
from craft_text_detector.file_utils import rectify_poly
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import pandas as pd
import cv2
import torch
import os

# Fix PIL ANTIALIAS for newer Pillow versions
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

#sort chữ khi crop line để theo thứ tự từ trái sang phải
def sort_img(regions):
  for i in range(len(regions) - 1):
    min_idx = i
    for j in range(i, len(regions)):
      if abs(regions[min_idx][0, 1] - regions[j][0, 1]) > 10:
        if regions[min_idx][0, 1] > regions[j][0, 1]:
          min_idx = j
      else:
        if regions[min_idx][0, 0] > regions[j][0, 0]:
          min_idx = j
    regions[min_idx], regions[i] = regions[i], regions[min_idx]
  return regions

def ocr(i, detector):
  if i.shape[0] < i.shape[1]:
    img_pil = Image.fromarray(i)
    s, p = detector.predict(img_pil, return_prob = True)
    if p > 0.7:
        return (s, p)
    else:
        return (0, 0)
  else:
    im1 = Image.fromarray(i)
    s1, p1 = detector.predict(im1, return_prob = True)
    s, p = s1, p1

    im2 = cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE)
    s2, p2 = detector.predict(Image.fromarray(im2), return_prob = True)
    if p2 > p:
        p, s = p2, s2

    im3 = cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE)
    s3, p3 = detector.predict(Image.fromarray(im3), return_prob = True)
    if p3 > p:
        p, s = p3, s3
    if p > 0.7:
        return (s, p)
    else:
        return (0, 0)
    
def read(img, key, craft_net, refine_net, detector):
  image = read_image(img)
  is_cuda = torch.cuda.is_available()
                    
  #predict craft
  link_thresh = 0.3 if key == 0 else 0.1
  low_text_thresh = 0.3 if key == 0 else (0.05 if key == 3 else 0.2)
  
  prediction_result = get_prediction(
      image=image,
      craft_net=craft_net,
      refine_net=refine_net,
      text_threshold=0.7,
      link_threshold=link_thresh,
      low_text=low_text_thresh,
      cuda=is_cuda,
      long_size=1280
  )
  regions=prediction_result["polys"]
  sort_img(regions)

  a = [rectify_poly(image, i) for i in regions]
  p_total = 0
  s_total = ''
  for i in a:
    s_temp, p_temp = ocr(i, detector)
    if s_temp != 0:
      p_total += p_temp
      s_total += s_temp + " "
  
  if len(a) > 0:
    p_total = p_total / len(a)
    
  return s_total.strip(), p_total

def craft_and_ocr(results, fn):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    is_cuda = torch.cuda.is_available()
    
    # load models
    try:
        refine_net = load_refinenet_model(cuda=is_cuda)
    except:
        print("⚠️ refine_net load failed, using None")
        refine_net = None
        
    craft_net = load_craftnet_model(cuda=is_cuda)
    
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained']=True
    config['device'] = device
    config['predictor']['beamsearch']=False
    detector = Predictor(config)

    out = []
    idx = 0
    for info, cache in results:
        res = {0: "", 1: "", 2: "", 3: "", 4: "", 5: ""}
        for key, value in info.items():
            for img in value:
                if img.shape[0] < img.shape[1] * 2:
                    s, _ = read(img, key, craft_net, refine_net, detector)
                    res[key] += s + " "
                else:
                    s1, p1 = read(img, key, craft_net, refine_net, detector)
                    s, p = s1, p1
                    
                    im2 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    s2, p2 = read(im2, key, craft_net, refine_net, detector)
                    if p2 > p: p, s = p2, s2
                    
                    im3 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    s3, p3 = read(im3, key, craft_net, refine_net, detector)
                    if p3 > p: p, s = p3, s3
                    
                    res[key] += s + " "
        
        ten_sach = res[0]
        # Xử lý text chồng lấn
        for i in cache:
            target = res[i].strip()
            if not target: continue
            if target in ten_sach:
                ten_sach = ten_sach.replace(target, '')
            else:
                for word in target.split():
                    ten_sach = ten_sach.replace(word, '')
        
        out.append({
            'file names' : fn[idx],
            'Ten sach': ten_sach.strip(),
            'Tac gia': res[1].strip(),
            'Nha xuat ban': res[2].strip(),
            'Tap': res[3].strip(),
            'Nguoi dich': res[4].strip(),
            'Tai ban': res[5].strip()
        })
        idx += 1
        
    output = pd.DataFrame(out)
    if not output.empty:
        output = output.sort_values(by=['file names'])
    return output
