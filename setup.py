import os
import numpy
from   setuptools import setup
from   torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

nvjpeg_found = CUDA_HOME is not None and os.path.exists(os.path.join(CUDA_HOME, "include", "nvjpeg.h"))

print(f'NVJPEG Found or Not: {nvjpeg_found}')

# python setup.py build_ext
setup(
    name         = "PyNvJpeg",
    version      = "1.0.0",
    author       = "yangxian",
    author_email = "yangxian-001@cpic.com.cn",
    packages     = ['nvjpeg'],
    ext_modules  = [
        CUDAExtension(name         = "nvjpeg_wrapper", 
                      include_dirs = [numpy.get_include(), ],
                      sources      = ['src/wrapper.cpp'], 
                      libraries    = ['nvjpeg']
                    )
    ],
    cmdclass     = {
        "build_ext" : BuildExtension
    }
)

