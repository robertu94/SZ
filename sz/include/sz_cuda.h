#ifndef SZ_CUDA_H
#define SZ_CUDA_H
void
calculate_regression_coefficents_host(
        const float* oriData,
        struct sz_opencl_sizes const* sizes,
        float* reg_params,
        float* const pred_buffer);
#endif
