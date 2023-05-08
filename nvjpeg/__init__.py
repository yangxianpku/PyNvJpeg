import ctypes
import torch
import numpy
from   typing import Tuple, Union

import nvidia.dali                as     dali
import nvidia.dali.types          as     types
from   nvidia.dali.backend        import TensorGPU, TensorCPU
from   nvidia.dali.plugin.pytorch import to_torch_type
from   nvjpeg_wrapper             import (NvJpegExecption, NvJpeg, ChromaSubsampling,
                                          InputFormat, OutputFormat)

__version__    = "1.0.0"
__author__     = "xyswjtu@163.com"

exception      = NvJpegExecption

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


class nvjpegChromaSubsampling:            # 编码时默认的下采样格式，默认为CSS_420(与OpenCV保持一致)
    CSS_444    = ChromaSubsampling.CSS_444
    CSS_422    = ChromaSubsampling.CSS_422
    CSS_420    = ChromaSubsampling.CSS_420
    CSS_440    = ChromaSubsampling.CSS_440
    CSS_411    = ChromaSubsampling.CSS_411
    CSS_410    = ChromaSubsampling.CSS_410
    CSS_GRAY   = ChromaSubsampling.CSS_GRAY
    CSS_410V   = ChromaSubsampling.CSS_410V
    CSS_UNK    = ChromaSubsampling.CSS_UNK


class nvjpegInputFormat:            # 指定编码时输入的数据格式
    RGBP       = InputFormat.RGB      # 1.输入格式为CHW格式的RGB数据
    BGRP       = InputFormat.BGR      # 2.输入格式为CHW格式的BGR数据
    RGB        = InputFormat.RGBI     # 3.输入格式为HWC格式的RGB数据
    BGR        = InputFormat.BGRI     # 4.输入格式为HWC格式的BGR数据(默认)


class nvjpegOutputFormat:           # 指定解码时输出的数据格式
    RGBP       = OutputFormat.RGB     # 1.输出格式为CHW格式的RGB数据   
    BGRP       = OutputFormat.BGR     # 2.输出格式为CHW格式的BGR数据  
    RGB        = OutputFormat.RGBI    # 3.输出格式为HWC格式的RGB数据
    BGR        = OutputFormat.BGRI    # 4.输出格式为HWC格式的BGR数据(默认) 


class JpegCoder():
    """
    Summary: 
        JpegCoder类使用Pybind11对nvjpeg库(https://docs.nvidia.com/cuda/nvjpeg/index.html)进行了一定程度的封装, 主要封装
        了其编码和解码的功能。为了OpenCV使用者保持习惯, 该类的成员方法名均与OpenCV一致.

        [注意]: nvjpeg只支持对JPEG标准的图像解码, 图像格式为.jpg和.jpeg. 因此该工具的解码器部分封装了NVIDAI DALI工具, 其对常见
        的图像格式支持更加健全(支持JPG, BMP, PNG, TIFF, PNM, PPM, PGM, PBM, JPEG 2000, WebP等格式). 对于JPEG,JPEG2000和
        TIFF格式的图像, 其使用GPU加速的形式进行解码,对于其它格式回退使用OpenCV进行解码.
    """
    def __init__(self) -> None:
        self.nvjpeg_coder = NvJpeg()
        self._bgri_pipe = dali_decoder_bgri_pipe()
        self._rgbi_pipe = dali_decoder_rgbi_pipe()
        self._bgrp_pipe = dali_decoder_bgrp_pipe()
        self._rgbp_pipe = dali_decoder_rgbp_pipe()
        self._bgri_pipe.build()
        self._rgbi_pipe.build()
        self._bgrp_pipe.build()
        self._rgbp_pipe.build()


    def imread(self, filename : str,
               format : nvjpegOutputFormat = nvjpegOutputFormat.BGR) -> torch.Tensor:
        """
        Summary: 读取图像数据并解码为torch.Tensor(GPU, torch.uint8).

        Args:
            filename: str,                            图像数据路径.
            format:   nvjpegOutputFormat              解码后的图像数据格式, 仅支持nvjpegOutputFormat的四种格式.
            
        Returns:
            image:    torch.Tensor(GPU, torch.uint8), 解码后的图像数据张量, 解码失败返回None.
        """
        buffer = numpy.fromfile(filename, dtype=numpy.uint8)
        image  = None
        if format == nvjpegOutputFormat.BGR:
            self._bgri_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            image, _ = self._bgri_pipe.run()
        elif format == nvjpegOutputFormat.RGB:
            self._rgbi_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            image, _ = self._rgbi_pipe.run()
        elif format == nvjpegOutputFormat.BGRP:
            self._bgrp_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            image, _ = self._bgrp_pipe.run()
        elif format == nvjpegOutputFormat.RGBP:
            self._rgbp_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            image, _ = self._rgbp_pipe.run()
        return None if image is None else to_tensor(image[0]) # 其它输出格式或解码失败


    def imdecode(self, data : Union[bytes, numpy.ndarray], 
                format : nvjpegOutputFormat = nvjpegOutputFormat.BGR) -> torch.Tensor:
        """
        Summary: 读取图像数据buffer并解码为torch.Tensor(GPU, torch.uint8).

        Args:
            data:     
                type = bytes                          解码前的图像数据bytes.
                type = numpy.ndarray                  解码前的图像数据,须为numpy.uint8类型.
            format:   nvjpegOutputFormat              解码后的图像数据格式, 仅支持nvjpegOutputFormat的四种格式.
            
        Returns:
            image:    torch.Tensor(GPU, torch.uint8), 解码后的图像数据张量.
        """
        if isinstance(data, bytes):
            buffer = numpy.frombuffer(data, dtype=numpy.uint8)
        else:
            buffer = data
            assert buffer.dtype == numpy.uint8, f"numpy ndarray expected uint8 but got {buffer.dtype}"
        image  = None
        if format == nvjpegOutputFormat.BGR:
            self._bgri_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            image, _ = self._bgri_pipe.run()
        elif format == nvjpegOutputFormat.RGB:
            self._rgbi_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            image, _ = self._rgbi_pipe.run()
        elif format == nvjpegOutputFormat.BGRP:
            self._bgrp_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            image, _ = self._bgrp_pipe.run()
        elif format == nvjpegOutputFormat.RGBP:
            self._rgbp_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            image, _ = self._rgbp_pipe.run()
        return None if image is None else to_tensor(image[0])  # 其它输出格式或解码失败


    def imencode(self, 
                 data        : torch.Tensor,
                 quality     : int                             = 75,
                 format      : nvjpegInputFormat               = nvjpegInputFormat.BGR,
                 subsampling : nvjpegChromaSubsampling         = nvjpegChromaSubsampling.CSS_420) -> torch.Tensor:
        """
        Summary: 将GPU Torch Tensor图像数据编码且写到指定的路径.

        Args:
            data:        type = torch.Tensor(GPU, torch.uint8)  需编码保存的图像数据.                                    
            quality:     int(0~100)                             编码压缩时的质量控制,值越大压缩比越小,图像质量越高.
            format:      nvjpegInputFormat                      待编码保存的图像数据的格式, 仅支持nvjpegInputFormat中的4种格式.
            subsampling: nvjpegChromaSubsampling                编码压缩时的YUV下采样率.
            
        Returns:
            buffer: torch.Tensor(CPU), 图像编码后的buffer字节.
        """
        assert quality > 0 and quality <= 100, "quality should be between 0 and 100"
        return self.nvjpeg_coder.imencode(data, quality, format, subsampling)


    def imwrite(self,
                filename    : str, 
                data        : torch.Tensor,
                quality     : int                             = 75, 
                format      : nvjpegInputFormat               = nvjpegInputFormat.BGR,
                subsampling : nvjpegChromaSubsampling         = nvjpegChromaSubsampling.CSS_420) -> bool:
        """
        Summary: 将GPU Torch Tensor图像数据编码且写到指定的路径.

        Args:
            filename:    str,                             图像的保存路径.
            data:        
                type = torch.Tensor(GPU, torch.uint8),    需编码保存的图像数据.
                #type = int(GPU数据的指针),  可通过Torch和DALI的data_ptr()方法获得
            shape:       Tuple[int]                       如果传入的data是指针类型,则必须指定图像的shape信息
            quality:     int(0~100),                      编码压缩时的质量控制,值越大压缩比越小,图像质量越高.
            format:      nvjpegInputFormat                待编码保存的图像数据的格式, 仅支持nvjpegInputFormat中的4种格式.
            subsampling: nvjpegChromaSubsampling                编码压缩时的YUV下采样率.
            
        Returns:
            success: bool, 图像是否编码且保存成功.
        """
        assert quality > 0 and quality <= 100, "quality should be between 0 and 100"
        return self.nvjpeg_coder.imwrite(filename, data, quality, format, subsampling)


    def shape(self, data : Union[str, torch.Tensor, numpy.ndarray]) -> Tuple[int]:
        """
        Summary: 获取图像的形状信息，格式为(H, W, C), C通常为3.

        Args:
            data: 
                type = str:            图像文件的路径字符串表示;
                type = torch.Tensor:   图像字节buffer的torch张量, 须为一维CPU Tensor, 且dtype==torch.uint8;
                type = numpy.ndarray:  图像字节buffer的numpy数组, 须为一维ndarray, 且dtype==numpy.uint8;
            
        Returns:
            shape: Tuple, 图像的形状信息，格式为(H, W, C), C通常为3.
        """
        if isinstance(data, str):
            return self.nvjpeg_coder.shape_filename(data)
        elif isinstance(data, numpy.ndarray):
            return self.nvjpeg_coder.shape_numpy(data)
        else:
            return self.nvjpeg_coder.shape_torch(data)


    def version(self) -> str:
        """
        Summary: 获取nvjepg的版本信息.

        Args:
            none.
            
        Returns:
            version: str, nvjpeg的版本信息.
        """
        return self.nvjpeg_coder.version()

    def __repr__(self) -> str:
        return self.nvjpeg_coder.__repr__()
