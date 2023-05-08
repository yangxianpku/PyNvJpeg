import cv2
import torch
import nvjpeg
import numpy as np



img_path = "../images/proposer.jpg"
# img_path = "../images/png.jpeg"


###########################################################################
# 1. OpenCV Decoder
print("cv2 shape: ", cv2.imread(img_path, cv2.IMREAD_UNCHANGED).shape)


###########################################################################
# 2. NvJpeg
nj   = nvjpeg.JpegCoder()

print(nj.shape(img_path))

img_np      = np.fromfile(img_path, dtype=np.uint8)
print(nj.shape(img_np))

img_torch   = torch.from_numpy(img_np)
print(nj.shape(img_torch))

