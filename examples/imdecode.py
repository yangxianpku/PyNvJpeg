import cv2
import torch
import nvjpeg
import ctypes
import numpy as np

img_path = "../images/proposer.jpg"

img_np = np.fromfile(img_path, dtype=np.uint8)

############################################################
# bgr测试
img_cv2_bgr   = cv2.imread(img_path,      cv2.IMREAD_COLOR)
img_cv2_rgb   = cv2.cvtColor(img_cv2_bgr, cv2.COLOR_BGR2RGB)
img_cv2_bgrp  = np.transpose(img_cv2_bgr, (2,0,1))
img_cv2_rgbp  = np.transpose(img_cv2_rgb, (2,0,1))

with open(img_path, 'rb') as f:
    # 读取文件内容
    image_bytes = f.read()

nj   = nvjpeg.JpegCoder()

# data bytes as input
# img_nj_bgr   = nj.imdecode(image_bytes, format=nvjpeg.nvjpegOutputFormat.BGR)
# img_nj_rgb   = nj.imdecode(image_bytes, format=nvjpeg.nvjpegOutputFormat.RGB)
# img_nj_bgrp  = nj.imdecode(image_bytes, format=nvjpeg.nvjpegOutputFormat.BGRP)
# img_nj_rgbp  = nj.imdecode(image_bytes, format=nvjpeg.nvjpegOutputFormat.RGBP)
# print(img_nj_bgr.shape, img_nj_rgb.shape, img_nj_bgrp.shape, img_nj_rgbp.shape)

# numpy ndarray as input
# img_np = img_np.astype(np.float32)
img_nj_bgr   = nj.imdecode(img_np, format=nvjpeg.nvjpegOutputFormat.BGR)
img_nj_rgb   = nj.imdecode(img_np, format=nvjpeg.nvjpegOutputFormat.RGB)
img_nj_bgrp  = nj.imdecode(img_np, format=nvjpeg.nvjpegOutputFormat.BGRP)
img_nj_rgbp  = nj.imdecode(img_np, format=nvjpeg.nvjpegOutputFormat.RGBP)
print(img_nj_bgr.shape, img_nj_rgb.shape, img_nj_bgrp.shape, img_nj_rgbp.shape)