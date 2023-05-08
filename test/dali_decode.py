import cv2
import torch
import ctypes
import numpy as np

import nvidia.dali                as dali
import nvidia.dali.types          as types
from   nvidia.dali.backend        import TensorGPU, TensorCPU
from   nvidia.dali.plugin.pytorch import to_torch_type

img_path = "../images/proposer.jpg"
# img_path = "../images/png.jpeg"

def to_tensor(dali_tensor, cuda_stream=None):
    """
    将DALI Tensor拷贝为PyTorch的Tensor.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU或nvidia.dali.backend.TensorGPU类型的张量, 注意不能为TensorList
    `cuda_stream` : torch.cuda.Stream, cudaStream_t 或者其他可以被转换为cudaStream_t类型的值.

    Returns:
    `pytorch.Tensor` 返回Tensor的device属性取决于输入的dali_tensor, 如果dali_tensor是gpu的, 则返回的torch.Tensor也在
                GPU上, 否则在cpu上.
    ----------
    """
    torch_type = to_torch_type[dali_tensor.dtype]

    assert isinstance(dali_tensor, (TensorGPU, TensorCPU)),  \
                    "to_tensor仅支持Tensor形式(TensorGPU, TensorCPU), 不支持TensorList."
    
    if isinstance(dali_tensor, TensorGPU):
        cuda_stream = types._raw_cuda_stream(cuda_stream)
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        tensor = torch.empty(dali_tensor.shape(), dtype=torch_type, device="cuda")
        c_type_pointer = ctypes.c_void_p(tensor.data_ptr())
        dali_tensor.copy_to_external(c_type_pointer, stream, non_blocking=True)
        return tensor
    else:
        tensor = torch.empty(dali_tensor.shape(), dtype=torch_type, device="cpu")
        c_type_pointer = ctypes.c_void_p(tensor.data_ptr())
        dali_tensor.copy_to_external(c_type_pointer)
        return tensor


@dali.pipeline_def(batch_size=1, num_threads=4, device_id=0, exec_pipelined=False, exec_async=False)
def dali_decoder_bgri_pipe():
    images      = dali.fn.external_source(device="cpu", name="bytes")
    images      = dali.fn.decoders.image(images, device="mixed",
                                         output_type=types.DALIImageType.BGR, use_fast_idct=True)
    shape       = dali.fn.shapes(images, dtype=types.INT32)
    return images, shape


@dali.pipeline_def(batch_size=1, num_threads=4, device_id=0, exec_pipelined=False, exec_async=False)
def dali_decoder_rgbi_pipe():
    images      = dali.fn.external_source(device="cpu", name="bytes")
    images      = dali.fn.decoders.image(images, device="mixed",
                                         output_type=types.DALIImageType.RGB, use_fast_idct=True)
    shape       = dali.fn.shapes(images, dtype=types.INT32)
    return images, shape


@dali.pipeline_def(batch_size=1, num_threads=4, device_id=0, exec_pipelined=False, exec_async=False)
def dali_decoder_bgrp_pipe():
    images      = dali.fn.external_source(device="cpu", name="bytes")
    images      = dali.fn.decoders.image(images, device="mixed",
                                         output_type=types.DALIImageType.BGR, use_fast_idct=True)
    images      = dali.fn.transpose(images, perm=[2, 0, 1])
    shape       = dali.fn.shapes(images, dtype=types.INT32)
    return images, shape


@dali.pipeline_def(batch_size=1, num_threads=4, device_id=0, exec_pipelined=False, exec_async=False)
def dali_decoder_rgbp_pipe():
    images      = dali.fn.external_source(device="cpu", name="bytes")
    images      = dali.fn.decoders.image(images, device="mixed",
                                         output_type=types.DALIImageType.RGB, use_fast_idct=True)
    images      = dali.fn.transpose(images, perm=[2, 0, 1])
    shape       = dali.fn.shapes(images, dtype=types.INT32)
    return images, shape



img_np = np.fromfile(img_path, dtype=np.uint8)

############################################################
# bgr测试
img_cv2_bgr   = cv2.imread(img_path,      cv2.IMREAD_COLOR)
img_cv2_rgb   = cv2.cvtColor(img_cv2_bgr, cv2.COLOR_BGR2RGB)
img_cv2_bgrp  = np.transpose(img_cv2_bgr, (2,0,1))
img_cv2_rgbp  = np.transpose(img_cv2_rgb, (2,0,1))

# bgri_pipe    = dali_decoder_bgri_pipe()
# bgri_pipe.build()
# bgri_pipe.feed_input("bytes", np.expand_dims(img_np, axis=0))
# image, shape = bgri_pipe.run()
# img_dali_bgr = to_tensor(image[0]).cpu().numpy()

# print(img_cv2_bgr)
# print("=================================")
# print(img_dali_bgr)

# rgbi_pipe    = dali_decoder_rgbi_pipe()
# rgbi_pipe.build()
# rgbi_pipe.feed_input("bytes", np.expand_dims(img_np, axis=0))
# image, shape = rgbi_pipe.run()
# img_dali_rgb = to_tensor(image[0]).cpu().numpy()
# print(img_cv2_rgb)
# print("=================================")
# print(img_dali_rgb)

# bgrp_pipe    = dali_decoder_bgrp_pipe()
# bgrp_pipe.build()
# bgrp_pipe.feed_input("bytes", np.expand_dims(img_np, axis=0))
# image, shape = bgrp_pipe.run()
# img_dali_bgrp = to_tensor(image[0]).cpu().numpy()
# print(img_cv2_bgrp)
# print("=================================")
# print(img_dali_bgrp)

# rgbp_pipe    = dali_decoder_rgbp_pipe()
# rgbp_pipe.build()
# rgbp_pipe.feed_input("bytes", np.expand_dims(img_np, axis=0))
# image, shape = rgbp_pipe.run()
# img_dali_rgbp = to_tensor(image[0]).cpu().numpy()
# print(img_cv2_rgbp)
# print("=================================")
# print(img_dali_rgbp)


rgbp_pipe     = dali_decoder_rgbp_pipe()
rgbp_pipe.build()
rgbp_pipe.feed_input("bytes", np.expand_dims(img_np, axis=0))
image, shape  = rgbp_pipe.run()
img_dali_rgbp = to_tensor(image[0])
img_dali_rgbp_cpu = to_tensor(image[0].as_cpu())

############################################################
print(img_dali_rgbp.shape, img_dali_rgbp.dtype, img_dali_rgbp.device)
print(img_dali_rgbp_cpu.shape, img_dali_rgbp_cpu.dtype, img_dali_rgbp_cpu.device)
