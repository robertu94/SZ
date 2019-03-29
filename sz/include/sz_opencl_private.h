/**
 * this file is for any non-public methods or structures used by the sz_opencl implementation
 * that do not also need to be included on a device kernel
 */
#include <string>
#include "sz_opencl_config.h"

#if !SZ_OPENCL_USE_CUDA
#include <CL/cl.hpp>
#endif

struct sz_opencl_state
{
  struct error_t
  {
    int code = 0;
    std::string str;
  } error;
#if !SZ_OPENCL_USE_CUDA
  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;
  cl::Kernel calculate_regression_coefficents;
#endif
  int debug_level = 0;
};

struct sz_opencl_coefficient_params {
  sz_opencl_coefficient_params(cl_ulong reg_count, cl_ulong num_blocks, cl_float* reg_params,cl_int* coeff_result_type, cl_float* coeff_unpredictable_data):
    last_coeffcients{ 0.0, 0.0, 0.0, 0.0 },
    coeff_result_type{coeff_result_type},
    coeff_unpredicatable_data{coeff_unpredictable_data},
    coeff_type{},
    coeff_unpred_data{},
    coeff_unpredictable_count{0,0,0,0},
    reg_params_separte{}
  {
    for (int i = 0; i < 4; i++) {
      coeff_type[i] = coeff_result_type + i * reg_count;
      coeff_unpred_data[i] = coeff_unpredictable_data + i * reg_count;
      reg_params_separte[i] = reg_params + i * num_blocks;
    }
  }

    cl_float last_coeffcients[4];
    cl_int* coeff_result_type;
    cl_float* coeff_unpredicatable_data;
    cl_int* coeff_type[4];
    cl_float* coeff_unpred_data[4];
    cl_uint coeff_unpredictable_count[4];
    cl_float* reg_params_separte[4];
};
