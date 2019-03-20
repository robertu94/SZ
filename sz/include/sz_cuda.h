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
#endif
