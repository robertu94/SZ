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

void save_unpredictable_body_host(const sz_opencl_sizes *sizes,
                                  double realPrecision,
                                  float mean,
                                  bool use_mean,
                                  int intvRadius,
                                  int intvCapacity,
                                  int intvCapacity_sz,
                                  const float *reg_params,
                                  const unsigned char *indicator,
                                  const unsigned long *reg_params_pos_index,
                                  float *data_buffer,
                                  int *blockwise_unpred_count,
                                  float *unpredictable_data,
                                  int *result_type);

void decompress_all_blocks_host(const sz_opencl_sizes* sizes,
                                double realPrecision, float mean, unsigned char use_mean,
                                const unsigned char* indicator, const float* reg_params,
                                int intvRadius, const size_t* unpred_offset,
                                const float* unpred_data,
                                const sz_opencl_decompress_positions* pos,
                                const int* result_type,
                                float* dec_block_data, size_t data_unpred_size);
#endif
