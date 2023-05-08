import cv2
import time
import torch
import nvjpeg
import numpy as np

img_path = "../images/proposer.jpg"


img_cv2_bgr   = cv2.imread(img_path,      cv2.IMREAD_COLOR)
img_cv2_rgb   = cv2.cvtColor(img_cv2_bgr, cv2.COLOR_BGR2RGB)
img_cv2_bgrp  = np.transpose(img_cv2_bgr, (2,0,1))
img_cv2_rgbp  = np.transpose(img_cv2_rgb, (2,0,1))

nj   = nvjpeg.JpegCoder()

img_nj_bgr  = nj.imread(img_path, format=nvjpeg.nvjpegOutputFormat.BGR)
print(img_cv2_bgr)
print("============================")
print(img_nj_bgr)


img_nj_rgb  = nj.imread(img_path, format=nvjpeg.nvjpegOutputFormat.RGB)
print(img_cv2_rgb)
print("============================")
print(img_nj_rgb)


img_nj_bgrp = nj.imread(img_path, format=nvjpeg.nvjpegOutputFormat.BGRP)
print(img_cv2_bgrp)
print("============================")
print(img_nj_bgrp)

img_nj_rgbp = nj.imread(img_path, format=nvjpeg.nvjpegOutputFormat.RGBP)
print(img_cv2_rgbp)
print("============================")
print(img_nj_rgbp)
