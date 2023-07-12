import re
import nvjpeg
import struct
import cv2, base64
import io, os, signal, traceback
from   xml.etree import ElementTree

import numpy                as np
import nvidia.dali          as dali
import nvidia.dali.fn       as fn
import nvidia.dali.types    as types

from   PIL                  import Image, ImageFile
from   model.logger         import logger

ImageFile.LOAD_TRUNCATED_IMAGES = True                 # enable PIL对截断的图像也能解码
Image.MAX_IMAGE_PIXELS          = None                 # enable PIL解码时对图像的大小无限制
big_img_pixel                   = 16000000             # 大于4k*4k的图像使用CPU解码


# 1. BGR decoder pipe, HWC
@dali.pipeline_def(batch_size=1, num_threads=1, device_id=0, exec_pipelined=False, exec_async=False)
def img_decode_pipe():
    image       = dali.fn.external_source(device="cpu", name="bytes")
    image       = dali.fn.decoders.image(image, device="mixed",
                                         output_type=types.DALIImageType.BGR, use_fast_idct=True)
    shape       = fn.shapes(image, dtype=types.INT32)
    return image, shape


# 2. RGB decoder pipe, HWC
# @dali.pipeline_def(batch_size=1, num_threads=1, device_id=0, exec_pipelined=False, exec_async=False)
# def img_decode_pipe():
#     images      = dali.fn.external_source(device="cpu", name="bytes")
#     images      = dali.fn.decoders.image(images, device="mixed",
#                                          output_type=types.DALIImageType.RGB, use_fast_idct=True)
#     shape       = dali.fn.shapes(images, dtype=types.INT32)
#     return images, shape


# 3. BGR decoder pipe, CHW
# @dali.pipeline_def(batch_size=1, num_threads=1, device_id=0, exec_pipelined=False, exec_async=False)
# def img_decode_pipe():
#     images      = dali.fn.external_source(device="cpu", name="bytes")
#     images      = dali.fn.decoders.image(images, device="mixed",
#                                          output_type=types.DALIImageType.BGR, use_fast_idct=True)
#     images      = dali.fn.transpose(images, perm=[2, 0, 1])
#     shape       = dali.fn.shapes(images, dtype=types.INT32)
#     return images, shape


# 4. RGB decoder pipe, CHW
# @dali.pipeline_def(batch_size=1, num_threads=1, device_id=0, exec_pipelined=False, exec_async=False)
# def img_decode_pipe():
#     images      = dali.fn.external_source(device="cpu", name="bytes")
#     images      = dali.fn.decoders.image(images, device="mixed",
#                                          output_type=types.DALIImageType.RGB, use_fast_idct=True)
#     images      = dali.fn.transpose(images, perm=[2, 0, 1])
#     shape       = dali.fn.shapes(images, dtype=types.INT32)
#     return images, shape


class ImageCoder(object):
    def __init__(self) -> None:
        self._rotate_func = {
            1: lambda x: x,                                                             # 1：默认值，没有旋转，照片的方向是原始方向
            2: lambda x: cv2.flip(x, 1),                                                # 2：照片水平翻转
            3: lambda x: cv2.flip(x, -1),                                               # 3：顺时针旋转180度
            4: lambda x: cv2.flip(x, 0),                                                # 4：垂直翻转
            5: lambda x: cv2.transpose(x),                                              # 5：顺时针旋转90度并水平翻转
            6: lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),                        # 6：顺时针旋转90度
            7: lambda x: cv2.flip(cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE), 1),    # 7：逆时针旋转90度并水平翻转
            8: lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)                  # 8：逆时针旋转90度
        }

        self.pip_rebuildable      = True
        self.nj                   = nvjpeg.JpegCoder()
        self.img_decode_pipe      = img_decode_pipe()
        self.img_decode_pipe.build()
        logger.info("ImageCoder dali decoder pipe build success")


    def _get_exif_info(self, img_bytes):
        # 获取图像文件中exif信息头中的旋转信息
        try:
            exif_info   = Image.open(io.BytesIO(img_bytes))._getexif()
            return 1 if exif_info is None else exif_info.get(274, 1)
        except:
            return 1
 

    def _convertToPx(self, value):
        matched = re.match(r"(\d+)(?:\.\d)?([a-z]*)$", value)
        if not matched:
            raise ValueError("unknown length value: %s" % value)
        else:
            length, unit = matched.groups()
            if unit == "":
                return int(length)
            elif unit == "cm":
                return int(length) * 96 / 2.54
            elif unit == "mm":
                return int(length) * 96 / 2.54 / 10
            elif unit == "in":
                return int(length) * 96
            elif unit == "pc":
                return int(length) * 96 / 6
            elif unit == "pt":
                return int(length) * 96 / 6
            elif unit == "px":
                return int(length)
            else:
                raise ValueError("unknown unit type: %s" % unit)


    def _get_img_size(self, fhandle):
        """
        Return (width, height) for a given img file content
        no requirements
        :type filepath: Union[str, pathlib.Path]
        :rtype Tuple[int, int]
        """
        height = -1
        width  = -1

        head   = fhandle.read(24)
        size   = len(head)
        # handle GIFs
        if size >= 10 and head[:6] in (b'GIF87a', b'GIF89a'):
            # Check to see if content_type is correct
            try:
                width, height = struct.unpack("<hh", head[6:10])
            except struct.error:
                raise ValueError("Invalid GIF file")
        # see png edition spec bytes are below chunk length then and finally the
        elif size >= 24 and head.startswith(b'\211PNG\r\n\032\n') and head[12:16] == b'IHDR':
            try:
                width, height = struct.unpack(">LL", head[16:24])
            except struct.error:
                raise ValueError("Invalid PNG file")
        # Maybe this is for an older PNG version.
        elif size >= 16 and head.startswith(b'\211PNG\r\n\032\n'):
            # Check to see if we have the right content type
            try:
                width, height = struct.unpack(">LL", head[8:16])
            except struct.error:
                raise ValueError("Invalid PNG file")
        # handle JPEGs
        elif size >= 2 and head.startswith(b'\377\330'):
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf or ftype in [0xc4, 0xc8, 0xcc]:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except struct.error:
                raise ValueError("Invalid JPEG file")
        # handle JPEG2000s
        elif size >= 12 and head.startswith(b'\x00\x00\x00\x0cjP  \r\n\x87\n'):
            fhandle.seek(48)
            try:
                height, width = struct.unpack('>LL', fhandle.read(8))
            except struct.error:
                raise ValueError("Invalid JPEG2000 file")
        # handle big endian TIFF
        elif size >= 8 and head.startswith(b"\x4d\x4d\x00\x2a"):
            offset = struct.unpack('>L', head[4:8])[0]
            fhandle.seek(offset)
            ifdsize = struct.unpack(">H", fhandle.read(2))[0]
            for i in range(ifdsize):
                tag, datatype, count, data = struct.unpack(">HHLL", fhandle.read(12))
                if tag == 256:
                    if datatype == 3:
                        width = int(data / 65536)
                    elif datatype == 4:
                        width = data
                    else:
                        raise ValueError("Invalid TIFF file: width column data type should be SHORT/LONG.")
                elif tag == 257:
                    if datatype == 3:
                        height = int(data / 65536)
                    elif datatype == 4:
                        height = data
                    else:
                        raise ValueError("Invalid TIFF file: height column data type should be SHORT/LONG.")
                if width != -1 and height != -1:
                    break
            if width == -1 or height == -1:
                raise ValueError("Invalid TIFF file: width and/or height IDS entries are missing.")
        elif size >= 8 and head.startswith(b"\x49\x49\x2a\x00"):
            offset = struct.unpack('<L', head[4:8])[0]
            fhandle.seek(offset)
            ifdsize = struct.unpack("<H", fhandle.read(2))[0]
            for i in range(ifdsize):
                tag, datatype, count, data = struct.unpack("<HHLL", fhandle.read(12))
                if tag == 256:
                    width = data
                elif tag == 257:
                    height = data
                if width != -1 and height != -1:
                    break
            if width == -1 or height == -1:
                raise ValueError("Invalid TIFF file: width and/or height IDS entries are missing.")
        # handle SVGs
        elif size >= 5 and head.startswith(b'<?xml'):
            try:
                fhandle.seek(0)
                root   = ElementTree.parse(fhandle).getroot()
                width  = self._convertToPx(root.attrib["width"])
                height = self._convertToPx(root.attrib["height"])
            except Exception:
                raise ValueError("Invalid SVG file")

        return width, height


    def _cpu_decode(self, img_bytes, ignore_orientation = False, exif_info = 1):
        try:
            if ignore_orientation:
                image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            else:
                image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            assert image is not None
        except:
            image_content = Image.open(io.BytesIO(img_bytes))
            image_content = image_content.convert('RGB')
            image         = cv2.cvtColor(np.asarray(image_content), cv2.COLOR_RGB2BGR)

            if not ignore_orientation:
                image     = self._rotate_func.get(exif_info, self._rotate_func[1])(image)
            assert image is not None
        return image


    def imread(self, img_file, ignore_orientation = False):
        with open(img_file, "rb") as fp:     # TODO: 文件读取可以改成异步实现
            img_bytes = fp.read()
            return self.imdecode(img_bytes, ignore_orientation)


    def imdecode(self, img_bytes, ignore_orientation = False):
        exif_info = self._get_exif_info(img_bytes)

        if not self.pip_rebuildable:       # 1. 如果decode pipe损坏, 直接走cpu decode
            logger.warning("dali decoder pipe broken, use cpu for decoding")
            return self._cpu_decode(img_bytes, ignore_orientation, exif_info)

        # jpeg2000[CUDA 11.x]： img_bytes[:12] = b'\x00\x00\x00\x0cjP  \r\n\x87\n'
        if img_bytes[:2] != b'\xFF\xD8':   # 2. 如果不是jpeg图像, 直接走cpu decode
            logger.warning("not jpeg image, use cpu for decoding")
            return self._cpu_decode(img_bytes, ignore_orientation, exif_info)

        # 超大图形额外考虑
        try:
            w, h = self._get_img_size(io.BytesIO(img_bytes))
        except:
            logger.warning(f"unknown image size for: {traceback.format_exc()}")
            w, h = -1, -1

        if (w * h > big_img_pixel) or (w < 0 or h < 0):   # 3. 超大图像或者未获取到size的图像， 直接走CPU decode
            logger.warning(f"cpu decode for super large or unknown size of image!")
            return self._cpu_decode(img_bytes, ignore_orientation, exif_info)
        
        # 4. 常规size的jpeg图像优先尝试走gpu decode, 失败后走cpu decode
        try:
            self.img_decode_pipe.feed_input("bytes",
                                            np.expand_dims(np.frombuffer(img_bytes,
                                            dtype=np.uint8), axis=0))
            image, _  = self.img_decode_pipe.run()
            image     = image.as_cpu().as_array()[0]

            if not ignore_orientation:
                image = self._rotate_func.get(exif_info, self._rotate_func[1])(image)
        except:
            logger.warning(f"dali decode error, switch to cpu decoder, rebuild dali decoder pipe")
            try:
                del self.img_decode_pipe
                self.img_decode_pipe = img_decode_pipe()          # 解决dali管道失效的问题
                self.img_decode_pipe.build()
                logger.info("dali decoder pipe rebuild success")
            except:
                self.pip_rebuildable  = False
                logger.error(f"dali decoder pipe rebuild fail: {traceback.format_exc()}")
                # os.kill(os.getpid(), signal.SIGTERM)              # 直接杀死
            image   = self._cpu_decode(img_bytes, ignore_orientation, exif_info)
        return image


    def imwrite(self, img_path, img_tensor):
        # 图像编码，输入为unint的cuda torch tensor， HWC
        h, w = img_tensor.shape[:2]
        if h*w > big_img_pixel:
            logger.warning(f"super large image, switch to cv2 to imwrite!")
            return cv2.imwrite(img_path, img_tensor.cpu().numpy())

        try:
            return self.nj.imwrite(img_path, img_tensor)
        except:
            logger.warning(f"nvjpeg imwrite fail, switch to cv2: {traceback.format_exc()}")
            return cv2.imwrite(img_path, img_tensor.cpu().numpy())

    
    def imencode(self, img_tensor, to_base64 = True):
        # 图像编码，输入为unint的cuda torch tensor， HWC
        h, w = img_tensor.shape[:2]
        if h*w > big_img_pixel:
            logger.warning(f"super large image, switch to cv2 to encode!")
            _, buffer = cv2.imencode('.jpg', img_tensor.cpu().numpy())
            return str(base64.b64encode(buffer), encoding='utf-8') if to_base64 else buffer

        try:
            buffer = self.nj.imencode(img_tensor).numpy()
        except:
            logger.warning(f"nvjpeg encode fail, switch to cv2 for: {traceback.format_exc()}")
            _, buffer = cv2.imencode('.jpg', img_tensor.cpu().numpy())
        return str(base64.b64encode(buffer), encoding='utf-8') if to_base64 else buffer


# Notice
# 1. 注意大图显存溢出问题, 合理设置大图上限
# 2. 根据需要设置解码后的图像格式, 下采样方式，图像质量保留得分

# TODO:
# 1. GPU加速png与tiff解码;
# 2. 图像不解码resize操作;
