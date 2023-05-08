#include <vector>
#include <cassert>
#include <memory>
#include <iostream>

#include <nvjpeg.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/numpy.h>


inline const char* error_string(nvjpegStatus_t status) {
  switch(status) {
    case NVJPEG_STATUS_SUCCESS:                       return "success";
    case NVJPEG_STATUS_NOT_INITIALIZED:               return "not initialized";
    case NVJPEG_STATUS_INVALID_PARAMETER:             return "invalid parameter";
    case NVJPEG_STATUS_BAD_JPEG:                      return "bad jpeg";
    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:            return "jpeg not supported";
    case NVJPEG_STATUS_ALLOCATOR_FAILURE:             return "allocation failed";
    case NVJPEG_STATUS_EXECUTION_FAILED:              return "execution failed";
    case NVJPEG_STATUS_ARCH_MISMATCH:                 return "arch mismatch";
    case NVJPEG_STATUS_INTERNAL_ERROR:                return "internal error";
    case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:  return "implementation not supported";
    case NVJPEG_STATUS_INCOMPLETE_BITSTREAM:          return "incompatible bitstream";
    default:                                          return "unknown error";
  }
}


class NvJpegExecption: public std::exception {
public:
  NvJpegExecption(std::string const& ctx, nvjpegStatus_t status):  _status(status), _ctx(ctx) {}

  const char* what() const throw() {
    std::stringstream ss;
    ss << _ctx << ", nvjpeg error " << _status << ": " << error_string(_status);
    return ss.str().c_str();
  }

private:
  nvjpegStatus_t _status;
  std::string    _ctx;
};


inline void checkStatus(std::string const& ctx,
                      nvjpegStatus_t status) {
  if (nvjpegStatus_t::NVJPEG_STATUS_SUCCESS != status) {
    throw NvJpegExecption(ctx, status);
  }
}


class NvJpeg {
public:
  NvJpeg()  {
    nvjpegCreateSimple(&this->_handle);
    nvjpegJpegStateCreate(this->_handle,    &this->_status);
    nvjpegEncoderStateCreate(this->_handle, &this->_enc_state, NULL);
  }

  ~NvJpeg() {
    nvjpegEncoderStateDestroy(this->_enc_state);
    nvjpegJpegStateDestroy(this->_status);
    nvjpegDestroy(this->_handle);
  }

  const char* version(){
    std::stringstream ss;
    ss << NVJPEG_VER_MAJOR << "." << NVJPEG_VER_MINOR << "."  << NVJPEG_VER_PATCH << "." << NVJPEG_VER_BUILD;
    return ss.str().c_str();
  }

public:
    torch::Tensor imencode(torch::Tensor const& data,
                          int quality,
                          nvjpegInputFormat_t input_format, 
                          nvjpegChromaSubsampling_t subsampling) {
      py::gil_scoped_release release;
      size_t width, height;

      nvjpegEncoderParams_t params = this->_createParams(quality,
                                                        subsampling);
      nvjpegImage_t          image = this->_createEncoderImage(data,
                                                              input_format,
                                                              width,
                                                              height);

      checkStatus("nvjpegEncodeImage", 
                  nvjpegEncodeImage(this->_handle,
                  this->_enc_state,
                  params,
                  &image,
                  input_format,
                  width,
                  height,
                  nullptr
                ));
      
      size_t length;
      checkStatus("nvjpegEncodeRetrieveBitstream",
                  nvjpegEncodeRetrieveBitstream(this->_handle,
                  this->_enc_state,
                  NULL,
                  &length,
                  nullptr
                ));

      auto buffer = torch::empty({ int(length) },   // cpu buffer
                      torch::TensorOptions().dtype(torch::kUInt8));  

      checkStatus("nvjpegEncodeRetrieveBitstream",
                  nvjpegEncodeRetrieveBitstream(this->_handle,
                  this->_enc_state, 
                  (unsigned char*)buffer.data_ptr(),
                  &length, nullptr
                ));
      this->_destroyParams(params);

      return buffer;
    }


    // torch::Tensor imencodePtr(int offset,
    //                         //py::tuple shape, 
    //                         int quality,
    //                         nvjpegInputFormat_t input_format, 
    //                         nvjpegChromaSubsampling_t subsampling) {
    //   py::gil_scoped_release release;
    //   void* data_ptr = (void*)offset;

    //   int  height, width, channel;
    //   bool interleaved = input_format == NVJPEG_INPUT_RGBI || input_format == NVJPEG_INPUT_BGRI;
    //   // if (interleaved) { 
    //   //   height  = shape[0].cast<int>();
    //   //   width   = shape[1].cast<int>();
    //   //   channel = shape[2].cast<int>();
    //   // } else {
    //   //   channel = shape[0].cast<int>();
    //   //   height  = shape[1].cast<int>();
    //   //   width   = shape[2].cast<int>();
    //   // }
    //   height = 1100, width = 902, channel = 3;

    //   TORCH_CHECK(channel == 3,       "Channel must be equal to 3");

    //   nvjpegEncoderParams_t params = this->_createParams(quality, subsampling);
    //   nvjpegImage_t          image = this->_createEncoderImagePtr(data_ptr,
    //                                                               height,
    //                                                               width,
    //                                                               channel,
    //                                                               input_format);

    //   checkStatus("nvjpegEncodeImage",
    //               nvjpegEncodeImage(this->_handle,
    //                                 this->_enc_state,
    //                                 params,
    //                                 &image, 
    //                                 input_format, 
    //                                 width, 
    //                                 height, 
    //                                 nullptr
    //                               ));
      
    //   size_t length;
    //   checkStatus("nvjpegEncodeRetrieveBitstream",
    //               nvjpegEncodeRetrieveBitstream(this->_handle,
    //               this->_enc_state, 
    //               NULL, 
    //               &length, 
    //               nullptr
    //             ));
    //   auto buffer = torch::empty({ int(length) },    // cpu buffer
    //                             torch::TensorOptions().dtype(torch::kUInt8)); 

    //   checkStatus("nvjpegEncodeRetrieveBitstream",
    //             nvjpegEncodeRetrieveBitstream(this->_handle,
    //             this->_enc_state,
    //             (unsigned char*)buffer.data_ptr(),
    //             &length,
    //             nullptr)
    //           );
    //   this->_destroyParams(params);

    //   return buffer;
    // }


    bool imwrite(const std::string& filename, 
                torch::Tensor const& data,
                int quality,
                nvjpegInputFormat_t input_format,
                nvjpegChromaSubsampling_t subsampling) {
      auto buffer = this->imencode(data, quality, input_format, subsampling);
      TORCH_CHECK(buffer.device() == torch::kCPU, "Input tensor should be on CPU");
      TORCH_CHECK(buffer.dtype()  == torch::kU8,  "Input tensor dtype should be uint8");
      TORCH_CHECK(buffer.dim()    == 1,           "Input data should be a 1-dimensional tensor");

      auto fileBytes = buffer.data_ptr<uint8_t>();
      auto fileCStr  = filename.c_str();
      FILE* outfile  = fopen(fileCStr, "wb");

      TORCH_CHECK(outfile != nullptr, "Error opening output file");

      fwrite(fileBytes, sizeof(uint8_t), buffer.numel(), outfile);
      fclose(outfile);

      return true;
    }


    // bool imwritePtr(const std::string& filename, 
    //                 int offset, 
    //                 // py::tuple shape,
    //                 int quality,
    //                 nvjpegInputFormat_t input_format, 
    //                 nvjpegChromaSubsampling_t subsampling) {
    //   // void* data_ptr = PyCapsule_GetPointer(capsule, nullptr);
    //   auto buffer    = this->imencodePtr(offset, quality, input_format, subsampling);
    //   // auto buffer    = this->imencodePtr(data_ptr, shape, quality, input_format, subsampling);
      
    //   TORCH_CHECK(buffer.device() == torch::kCPU, "Input tensor should be on CPU");
    //   TORCH_CHECK(buffer.dtype()  == torch::kU8,  "Input tensor dtype should be uint8");
    //   TORCH_CHECK(buffer.dim()    == 1,           "Input data should be a 1-dimensional tensor");

    //   auto fileBytes = buffer.data_ptr<uint8_t>();
    //   auto fileCStr  = filename.c_str();
    //   FILE* outfile  = fopen(fileCStr, "wb");

    //   TORCH_CHECK(outfile != nullptr, "Error opening output file");

    //   fwrite(fileBytes, sizeof(uint8_t), buffer.numel(), outfile);
    //   fclose(outfile);

    //   return true;
    // }
    

    py::tuple  shape_torch(torch::Tensor const& data) {
      TORCH_CHECK(data.device() == torch::kCPU, "Input tensor should be on CPU");
      TORCH_CHECK(data.dtype()  == torch::kU8,  "Input tensor dtype should be uint8");
      TORCH_CHECK(data.dim()    == 1,           "Input data should be a 1-dimensional tensor");

      int widths[NVJPEG_MAX_COMPONENT]  = {0};
      int heights[NVJPEG_MAX_COMPONENT] = {0};
      int nComponents  = 0;
      nvjpegChromaSubsampling_t subsampling;
      nvjpegGetImageInfo(this->_handle, data.data_ptr<uint8_t>(), data.numel(),
                          &nComponents, &subsampling, widths, heights);

      return py::make_tuple(heights[0], widths[0], nComponents);
    }


    py::tuple  shape_numpy(py::array_t<uint8_t> data) {
      TORCH_CHECK(data.ndim()    == 1,           "Input data should be a 1-dimensional ndarray");

      int widths[NVJPEG_MAX_COMPONENT]  = {0};
      int heights[NVJPEG_MAX_COMPONENT] = {0};
      int nComponents  = 0;
      nvjpegChromaSubsampling_t subsampling;
      nvjpegGetImageInfo(this->_handle, data.mutable_data(), data.size(),
                          &nComponents, &subsampling, widths, heights);

      return py::make_tuple(heights[0], widths[0], nComponents);
    }


    py::tuple  shape_filename(const std::string& filename) {
      auto fileCStr = filename.c_str();
      FILE* infile  = fopen(fileCStr, "rb");

      TORCH_CHECK(infile != nullptr, "Error opening image file");

      fseek(infile, 0, SEEK_END);     
      long length = ftell(infile);


      unsigned char *buffer = (unsigned char*) malloc(length * sizeof(unsigned char));
      TORCH_CHECK(buffer != nullptr, "Error allocating buffer");

      rewind(infile);
      size_t read_bytes = fread(buffer, 1, length, infile);
      TORCH_CHECK(read_bytes == (size_t)length, "Error reading image file");
      fclose(infile);

      int widths[NVJPEG_MAX_COMPONENT]  = {0};
      int heights[NVJPEG_MAX_COMPONENT] = {0};
      int nComponents  = 0;
      nvjpegChromaSubsampling_t subsampling;
      nvjpegGetImageInfo(this->_handle, buffer, length,
                          &nComponents, &subsampling, widths, heights);

      free(buffer);
      return py::make_tuple(heights[0], widths[0], nComponents);
    }

private:
  inline nvjpegEncoderParams_t _createParams(int quality,
                                            nvjpegChromaSubsampling_t subsampling,
                                            int optimizedHuffman = 1,
                                            cudaStream_t stream = nullptr) {
    nvjpegEncoderParams_t params;
    nvjpegEncoderParamsCreate(this->_handle, &params, stream);
    nvjpegEncoderParamsSetQuality(params, quality, stream);      
    nvjpegEncoderParamsSetOptimizedHuffman(params, optimizedHuffman, stream);    
    nvjpegEncoderParamsSetSamplingFactors(params, subsampling, stream);  

    return params;
  }


  inline nvjpegStatus_t _destroyParams(nvjpegEncoderParams_t params) {
    return nvjpegEncoderParamsDestroy(params);
  }


  nvjpegImage_t _createEncoderImage(torch::Tensor const& data, nvjpegInputFormat_t input_format,
                                  size_t &width, size_t &height) const {
    TORCH_CHECK(data.is_cuda(),             "Input data should be on CUDA device");
    TORCH_CHECK(data.dtype() == torch::kU8, "Input data should be of uint8 dtype");
    TORCH_CHECK(data.is_contiguous(),       "Input data should be contiguous");

    nvjpegImage_t img; 
    for(int i = 0; i < NVJPEG_MAX_COMPONENT; i++){
      img.channel[i] = nullptr;
      img.pitch[i]   = 0;
    }

    bool interleaved = input_format == NVJPEG_INPUT_BGRI || input_format == NVJPEG_INPUT_RGBI;
    if (interleaved) {         // interleaved image, HWC
      TORCH_CHECK(data.dim() == 3 && data.size(2) == 3, "for interleaved (BGRI, RGBI) expected 3D tensor (H, W, C)");
      width  = data.size(1);
      height = data.size(0);   

      img.pitch[0]   = (unsigned int)at::stride(data, 0);
      img.channel[0] = (unsigned char*)data.data_ptr();
    } else {                   // planar image, CHW
      TORCH_CHECK(data.dim() == 3 && data.size(0) == 3, "for planar (BGR, RGB) expected 3D tensor (C, H, W)");
      width  = data.size(2);
      height = data.size(1); 

      size_t plane_stride = at::stride(data, 0);
      for(int i = 0; i < 3; i++) {
        img.pitch[i]   = (unsigned int)at::stride(data, 1);
        img.channel[i] = (unsigned char*)data.data_ptr() + plane_stride * i;
      }  
    }
    return img;
  }


  // nvjpegImage_t _createEncoderImagePtr(void* data_ptr, int height, int width, int channel = 3,
  //                               nvjpegInputFormat_t input_format = NVJPEG_INPUT_BGRI) const {
  //   nvjpegImage_t img; 
  //   for(int i = 0; i < NVJPEG_MAX_COMPONENT; i++) {
  //     img.channel[i] = nullptr;      
  //     img.pitch[i]   = 0;
  //   }

  //   bool interleaved = input_format == NVJPEG_INPUT_BGRI || input_format == NVJPEG_INPUT_RGBI;
  //   if (interleaved) {         // interleaved image      
  //     img.pitch[0]     = (unsigned int)(width * channel);
  //     img.channel[0]   = (unsigned char*)data_ptr;
  //   } else {                   // planar image
  //     size_t plane_stride = height * width;
  //     for(int i = 0; i < channel; i++) {
  //       img.pitch[i]   = (unsigned int)width;
  //       img.channel[i] = (unsigned char*)data_ptr + plane_stride * i;
  //     }  
  //   }
  //   return img;
  // }

private:
    nvjpegHandle_t         _handle;
    nvjpegJpegState_t      _status;
    nvjpegEncoderState_t   _enc_state;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {                                // TORCH_EXTENSION_NAME == nvjpeg_wrapper
  py::register_exception<NvJpegExecption>(m, "NvJpegExecption");          // 1. 注册异常处理类NvJpegExecption
  
  py::class_<NvJpeg>(m, "NvJpeg")                                         // 2. 注册NvJpeg类及相关函数
        .def(py::init<>())      // 构造函数
        .def("imencode",       &NvJpeg::imencode)                 
        .def("imwrite",        &NvJpeg::imwrite)    
        // .def("imencodePtr",    &NvJpeg::imencodePtr)                 
        // .def("imwritePtr",     &NvJpeg::imwritePtr)                
        .def("shape_torch",    &NvJpeg::shape_torch)
        .def("shape_numpy",    &NvJpeg::shape_numpy)
        .def("shape_filename", &NvJpeg::shape_filename)
        .def("version",        &NvJpeg::version)
        .def("__repr__", [](const NvJpeg &nvjpeg){ return "NvJpeg: python binding for nvjpeg at https://docs.nvidia.com/cuda/nvjpeg/index.html"; });

  py::enum_<nvjpegChromaSubsampling_t>(m, "ChromaSubsampling")            // 3. 注册采样枚举体
        .value("CSS_444",  nvjpegChromaSubsampling_t::NVJPEG_CSS_444)
        .value("CSS_422",  nvjpegChromaSubsampling_t::NVJPEG_CSS_422)
        .value("CSS_420",  nvjpegChromaSubsampling_t::NVJPEG_CSS_420)
        .value("CSS_440",  nvjpegChromaSubsampling_t::NVJPEG_CSS_440)
        .value("CSS_411",  nvjpegChromaSubsampling_t::NVJPEG_CSS_411)
        .value("CSS_410",  nvjpegChromaSubsampling_t::NVJPEG_CSS_410)
        .value("CSS_GRAY", nvjpegChromaSubsampling_t::NVJPEG_CSS_GRAY)
        .value("CSS_410V", nvjpegChromaSubsampling_t::NVJPEG_CSS_410V)
        .value("CSS_UNK",  nvjpegChromaSubsampling_t::NVJPEG_CSS_UNKNOWN)
        .export_values();

  py::enum_<nvjpegInputFormat_t>(m, "InputFormat")                        // 4. 注册编码时的输入图像数据格式
        .value("RGB",  nvjpegInputFormat_t::NVJPEG_INPUT_RGB)
        .value("BGR",  nvjpegInputFormat_t::NVJPEG_INPUT_BGR)
        .value("RGBI", nvjpegInputFormat_t::NVJPEG_INPUT_RGBI)
        .value("BGRI", nvjpegInputFormat_t::NVJPEG_INPUT_BGRI)
        .export_values();


  py::enum_<nvjpegOutputFormat_t>(m, "OutputFormat")                      // 5. 注册解码是的输出图像数据格式
        .value("RGB",       nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGB)               // RGB CHW
        .value("BGR",       nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGR)               // BGR CHW
        .value("RGBI",      nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGBI)              // RGB HWC
        .value("BGRI",      nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGRI)              // BGR HWC
        .export_values();
}