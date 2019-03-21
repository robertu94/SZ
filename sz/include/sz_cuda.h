#ifndef SZ_CUDA_H
#define SZ_CUDA_H
#include "sz_opencl_kernels.h"
void
calculate_regression_coefficents_host(
        const float* oriData,
        struct sz_opencl_sizes const* sizes,
        float* reg_params,
        float* const pred_buffer);

void copy_block_data_host(float **data,
                     const sz_opencl_decompress_positions &pos,
                     const float *dec_block_data);

void prepare_data_buffer_host(const float *oriData,
    sz_opencl_sizes const *sizes,
    cl_float *data_buffer);

void
opencl_sample_host(const sz_opencl_sizes* sizes,
              float mean,
              float noise,
              bool use_mean,
              const float* data_buffer,
              const float* reg_params_pos,
              unsigned char* indicator_pos
);
#endif
