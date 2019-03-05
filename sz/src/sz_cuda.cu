#include "cuda.h"
#include <iostream>
#include "cuda_runtime_api.h"
#include "sz_opencl_kernels.h"

__global__ void
calculate_regression_coefficents_kernel(
        const cl_float* oriData,
        struct sz_opencl_sizes const* sizes,
        float* reg_params,
        float* const pred_buffer)
{
    unsigned long i = threadIdx.x + blockIdx.x * blockDim.x;//get_global_id(0);
    unsigned long j = threadIdx.y + blockIdx.y * blockDim.y;//get_global_id(1);
    unsigned long k = threadIdx.z + blockIdx.z * blockDim.z;//get_global_id(2);
    if(i < sizes->num_x && j < sizes->num_y && j < sizes->num_z) {
    const unsigned int block_id =
        i * (sizes->num_y * sizes->num_z) + j * sizes->num_z + k;
    const float* local_data_pos = oriData + i * sizes->block_size * sizes->dim0_offset +
        j * sizes->block_size * sizes->dim1_offset + k * sizes->block_size;
    float* const pred_buffer_pos = pred_buffer+(block_id*sizes->num_blocks);
    for (size_t ii = 0; ii < sizes->block_size; ii++) {
        for (size_t jj = 0; jj < sizes->block_size; jj++) {
            for (size_t kk = 0; kk < sizes->block_size; kk++) {
                int ii_ = (i * sizes->block_size + ii < sizes->r1)
                    ? ii
                    : sizes->r1 - 1 - i * sizes->block_size;
                int jj_ = (j * sizes->block_size + jj < sizes->r2)
                    ? jj
                    : sizes->r2 - 1 - j * sizes->block_size;
                int kk_ = (k * sizes->block_size + kk < sizes->r3)
                    ? kk
                    : sizes->r3 - 1 - k * sizes->block_size;
                cl_ulong loc_data = ii_ * sizes->dim0_offset + jj_ * sizes->dim1_offset + kk_;
                cl_ulong loc_pred = ii * (sizes->block_size*sizes->block_size) + jj * sizes->block_size + kk;
                pred_buffer_pos[loc_pred] = local_data_pos[loc_data];
            }
        }
    }
    const float* cur_data_pos = pred_buffer+(block_id*sizes->num_blocks);
    float fx = 0.0;
    float fy = 0.0;
    float fz = 0.0;
    float f = 0;
    float sum_x, sum_y;
    float curData;
    for (size_t i = 0; i < sizes->block_size; i++) {
        sum_x = 0;
        for (size_t j = 0; j < sizes->block_size; j++) {
            sum_y = 0;
            for (size_t k = 0; k < sizes->block_size; k++) {
                //curData = *cur_data_pos;
                sum_y += curData;
                fz += curData * k;
                cur_data_pos++;
            }
            fy += sum_y * j;
            sum_x += sum_y;
        }
        fx += sum_x * i;
        f += sum_x;
    }
    float coeff =
        1.0 / (sizes->block_size * sizes->block_size * sizes->block_size);
    float* reg_params_pos = reg_params + block_id;
    reg_params_pos[0] = (2 * fx / (sizes->block_size - 1) - f) * 6 *
        coeff / (sizes->block_size + 1);
    reg_params_pos[sizes->params_offset_b] =
        (2 * fy / (sizes->block_size - 1) - f) * 6 * coeff /
        (sizes->block_size + 1);
    reg_params_pos[sizes->params_offset_c] =
        (2 * fz / (sizes->block_size - 1) - f) * 6 * coeff /
        (sizes->block_size + 1);
    reg_params_pos[sizes->params_offset_d] =
        f * coeff - ((sizes->block_size - 1) * reg_params_pos[0] / 2 +
                (sizes->block_size - 1) *
                reg_params_pos[sizes->params_offset_b] / 2 +
                (sizes->block_size - 1) *
                reg_params_pos[sizes->params_offset_c] / 2);
}
}

#define CUDA_SAFE_CALL( call) {                                    \
  cudaError err = call;                                                    \
  if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
        __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                  \
  } \
}

void
calculate_regression_coefficents_host(
        const cl_float* oriData,
        struct sz_opencl_sizes const* sizes,
        float* reg_params,
        float* const pred_buffer){
  int deviceNum;
  int maxBlockSize;
  cudaGetDevice(&deviceNum);
  cudaDeviceGetAttribute(&maxBlockSize, cudaDevAttrMaxThreadsPerBlock, deviceNum);
  maxBlockSize = floor(cbrt(maxBlockSize));


  dim3 block_size{maxBlockSize,maxBlockSize,maxBlockSize};
  dim3 grid_size{sizes->num_x/maxBlockSize + 1, sizes->num_y/maxBlockSize + 1, sizes->num_z/maxBlockSize + 1};

  float* oriData_d;
  struct sz_opencl_sizes* sizes_d;
  float* reg_params_d;
  float* pred_buffer_d;

  std::cout << "cuda" << std::endl;
  CUDA_SAFE_CALL(cudaMalloc(&oriData_d, sizeof(cl_float) * sizes->num_elements));
  CUDA_SAFE_CALL(cudaMalloc(&sizes_d, sizeof(sz_opencl_sizes)));
  CUDA_SAFE_CALL(cudaMalloc(&reg_params_d, sizeof(cl_float) * sizes->reg_params_buffer_size));
  CUDA_SAFE_CALL(cudaMalloc(&pred_buffer_d, sizeof(cl_float) * sizes->pred_buffer_size));
  CUDA_SAFE_CALL(cudaMemcpy(oriData_d, oriData, sizeof(cl_float) * sizes->num_elements, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(sizes_d, sizes, sizeof(struct sz_opencl_sizes), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemset(reg_params_d, 0, sizeof(cl_float) * sizes->reg_params_buffer_size));
  CUDA_SAFE_CALL(cudaMemset(pred_buffer_d, 0, sizeof(cl_float) * sizes->pred_buffer_size));

  calculate_regression_coefficents_kernel<<<grid_size, block_size>>>(
      oriData_d, sizes_d, reg_params, pred_buffer
      );

  CUDA_SAFE_CALL(cudaMemcpy(reg_params, reg_params_d, sizeof(cl_float) * sizes->reg_params_buffer_size, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(pred_buffer, pred_buffer_d, sizeof(cl_float) * sizes->pred_buffer_size, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(oriData_d));
  CUDA_SAFE_CALL(cudaFree(sizes_d));
  CUDA_SAFE_CALL(cudaFree(reg_params_d));
  CUDA_SAFE_CALL(cudaFree(pred_buffer_d));
}

