import cv2
import time
import torch
import nvjpeg
import numpy as np

img_path = "../images/proposer.jpg"

img_cv2_bgr     = cv2.imread(img_path,      cv2.IMREAD_COLOR)
img_cv2_rgb     = cv2.cvtColor(img_cv2_bgr, cv2.COLOR_BGR2RGB)
img_cv2_bgrp    = np.transpose(img_cv2_bgr, (2,0,1))
img_cv2_rgbp    = np.transpose(img_cv2_rgb, (2,0,1))

img_torch_bgr   = torch.from_numpy(img_cv2_bgr).to("cuda")
img_torch_rgb   = torch.from_numpy(img_cv2_rgb).to("cuda")
img_torch_bgrp  = torch.from_numpy(img_cv2_bgrp).to("cuda")
img_torch_rgbp  = torch.from_numpy(img_cv2_rgbp).to("cuda")


njp   = nvjpeg.JpegCoder()


# print(type(img_torch_bgr.data_ptr()))
# print(img_cv2_bgr.shape)

bytes_nj = njp.imencode(img_torch_bgr, format=nvjpeg.nvjpegInputFormat.BGR)
bytes_cv = cv2.imencode('.jpg', img_cv2_bgr)

print(bytes_nj)
print("=============================")
print(bytes_cv)