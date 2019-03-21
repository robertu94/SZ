#include "cuda.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include "cuda_runtime_api.h"
#include "sz_opencl_kernels.h"

#define CUDA_SAFE_CALL(call) {                                    \
  cudaError err = call;                                                    \
  if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
        __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                  \
  } \
}


#define CUDA_SAFE_KERNEL_CALL(call) {                                    \
	call; \
	cudaError_t err = cudaGetLastError(); \
  if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
        __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                  \
  } \
}

template <class  T, class U>
auto integer_divide_up(T a, U b) {
  size_t val = (a % b != 0) ? (a/b+1) : (a/b);
  if(val > std::numeric_limits<T>::max())
    throw std::domain_error("invalid integer division");
  else return val;
}


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
      const unsigned int block_id =
        i * (sizes->num_y * sizes->num_z) + j * sizes->num_z + k;
			if(block_id < sizes->num_blocks) {
      const float* cur_data_pos = pred_buffer + (block_id * sizes->max_num_block_elements);
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
            curData = *cur_data_pos;
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
      reg_params_pos[0] = (2 * fx / (sizes->block_size - 1) - f) * 6 * coeff /
                          (sizes->block_size + 1);
      reg_params_pos[sizes->params_offset_b] =
        (2 * fy / (sizes->block_size - 1) - f) * 6 * coeff /
        (sizes->block_size + 1);
      reg_params_pos[sizes->params_offset_c] =
        (2 * fz / (sizes->block_size - 1) - f) * 6 * coeff /
        (sizes->block_size + 1);
      reg_params_pos[sizes->params_offset_d] =
        f * coeff -
        ((sizes->block_size - 1) * reg_params_pos[0] / 2 +
         (sizes->block_size - 1) * reg_params_pos[sizes->params_offset_b] / 2 +
         (sizes->block_size - 1) * reg_params_pos[sizes->params_offset_c] / 2);
		}
}

void
calculate_regression_coefficents_host(
        const cl_float* oriData,
        struct sz_opencl_sizes const* sizes,
        float* reg_params,
        float* const pred_buffer){
  int deviceNum;
  unsigned int maxBlockSize;
  cudaGetDevice(&deviceNum);
  cudaDeviceGetAttribute((int*)&maxBlockSize, cudaDevAttrMaxThreadsPerBlock, deviceNum);
  maxBlockSize = floor(cbrt(maxBlockSize));


  dim3 block_size{maxBlockSize,maxBlockSize,maxBlockSize};
  dim3 grid_size{sizes->num_x/maxBlockSize + 1, sizes->num_y/maxBlockSize + 1, sizes->num_z/maxBlockSize + 1};

  float* oriData_d;
  struct sz_opencl_sizes* sizes_d;
  float* reg_params_d;
  float* pred_buffer_d;

  CUDA_SAFE_CALL(cudaMalloc(&oriData_d, sizeof(cl_float) * sizes->num_elements));
  CUDA_SAFE_CALL(cudaMalloc(&sizes_d, sizeof(sz_opencl_sizes)));
  CUDA_SAFE_CALL(cudaMalloc(&reg_params_d, sizeof(cl_float) * sizes->reg_params_buffer_size));
  CUDA_SAFE_CALL(cudaMalloc(&pred_buffer_d, sizeof(cl_float) * sizes->data_buffer_size));

  CUDA_SAFE_CALL(cudaMemcpy(oriData_d, oriData, sizeof(cl_float) * sizes->num_elements, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(sizes_d, sizes, sizeof(struct sz_opencl_sizes), cudaMemcpyHostToDevice));

  CUDA_SAFE_KERNEL_CALL((calculate_regression_coefficents_kernel<<<grid_size, block_size>>>(oriData_d, sizes_d, reg_params_d, pred_buffer_d)));

  CUDA_SAFE_CALL(cudaMemcpy(reg_params, reg_params_d, sizeof(cl_float) * sizes->reg_params_buffer_size, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(pred_buffer, pred_buffer_d, sizeof(cl_float) * sizes->data_buffer_size, cudaMemcpyDeviceToHost));

  CUDA_SAFE_CALL(cudaFree(oriData_d));
  CUDA_SAFE_CALL(cudaFree(sizes_d));
  CUDA_SAFE_CALL(cudaFree(reg_params_d));
  CUDA_SAFE_CALL(cudaFree(pred_buffer_d));
}

__global__
void copy_block_data_kernel(
    float* data,
    sz_opencl_decompress_positions const* pos,
    float const * dec_block_data
    ) {

  unsigned long i = threadIdx.x + blockIdx.x * blockDim.x;//get_global_id(0);
  unsigned long j = threadIdx.y + blockIdx.y * blockDim.y;//get_global_id(1);
  if(i < pos->data_elms1 && j < pos->data_elms2) {
    const float *block_data_pos =
        dec_block_data + (i + pos->resi_x) * pos->dec_block_dim0_offset + (j + pos->resi_y) * pos->dec_block_dim1_offset
            + pos->resi_z;
    float *final_data_pos = data + i * pos->data_elms2 * pos->data_elms3 + j * pos->data_elms3;
    for (cl_ulong k = 0; k < pos->data_elms3; k++) {
      *(final_data_pos++) = *(block_data_pos++);
    }
  }
}


void copy_block_data_host(float **data,
                     const sz_opencl_decompress_positions &pos,
                     const float *dec_block_data) {// extract data
  *data = (float*)malloc(sizeof(cl_float) * pos.data_buffer_size);

  float* data_d;
  float* dec_block_data_d;
  sz_opencl_decompress_positions* pos_d;

  int deviceNum;
  unsigned int maxBlockSize2;
  cudaGetDevice(&deviceNum);
  cudaDeviceGetAttribute((int*)&maxBlockSize2, cudaDevAttrMaxThreadsPerBlock, deviceNum);
  maxBlockSize2 = floor(sqrt(maxBlockSize2));


  dim3 block_size(maxBlockSize2,maxBlockSize2);
  dim3 grid_size(pos.data_elms1/maxBlockSize2+1, pos.data_elms2/maxBlockSize2+1);

  CUDA_SAFE_CALL(cudaMalloc(&data_d, sizeof(cl_float)* pos.data_buffer_size));
  CUDA_SAFE_CALL(cudaMalloc(&dec_block_data_d, sizeof(cl_float) *pos.dec_block_data_size));
  CUDA_SAFE_CALL(cudaMalloc(&pos_d, sizeof(sz_opencl_decompress_positions)));

  CUDA_SAFE_CALL(cudaMemcpy(pos_d, &pos, sizeof(struct sz_opencl_decompress_positions), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dec_block_data_d, dec_block_data, sizeof(cl_float)* pos.dec_block_data_size, cudaMemcpyHostToDevice));
  //do not copy data since we just malloc'ed it

  CUDA_SAFE_KERNEL_CALL((copy_block_data_kernel<<<grid_size,block_size>>>(data_d, pos_d, dec_block_data_d)));

  CUDA_SAFE_CALL(cudaMemcpy(*data, data_d, sizeof(cl_float) * pos.data_buffer_size, cudaMemcpyDeviceToHost));
  //do not copy sizes_d or pos_d, or dec_block_data_d back because they are const

  CUDA_SAFE_CALL(cudaFree(data_d));
  CUDA_SAFE_CALL(cudaFree(pos_d));
  CUDA_SAFE_CALL(cudaFree(dec_block_data_d));

}

__global__
void prepare_data_buffer_kernel(const float *oriData, const sz_opencl_sizes *sizes, cl_float *data_buffer) {
  unsigned long i = threadIdx.x + blockIdx.x * blockDim.x;//get_global_id(0);
  unsigned long j = threadIdx.y + blockIdx.y * blockDim.y;//get_global_id(1);
  unsigned long k = threadIdx.z + blockIdx.z * blockDim.z;//get_global_id(2);
  unsigned int block_id = i * (sizes->num_y * sizes->num_z) + j * sizes->num_z + k;

  if(block_id < sizes->num_blocks) {
    cl_float *data_buffer_location = data_buffer + block_id * sizes->max_num_block_elements;
    for (unsigned int ii = 0; ii < sizes->block_size; ii++) {
      for (unsigned int jj = 0; jj < sizes->block_size; jj++) {
        for (unsigned int kk = 0; kk < sizes->block_size; kk++) {
          // index in origin data
          cl_ulong i_ = i * sizes->block_size + ii;
          cl_ulong j_ = j * sizes->block_size + jj;
          cl_ulong k_ = k * sizes->block_size + kk;
          i_ = (i_ < sizes->r1) ? i_ : sizes->r1 - 1;
          j_ = (j_ < sizes->r2) ? j_ : sizes->r2 - 1;
          k_ = (k_ < sizes->r3) ? k_ : sizes->r3 - 1;
          data_buffer_location[ii * sizes->block_size * sizes->block_size + jj * sizes->block_size + kk] =
              oriData[i_ * sizes->r2 * sizes->r3 + j_ * sizes->r3 + k_];
        }
      }
    }
  }
}


void prepare_data_buffer_host(float const *oriData, sz_opencl_sizes const *sizes, cl_float *data_buffer) {
  float* oriData_d;
  sz_opencl_sizes* sizes_d;
  float* data_buffer_d;

  int deviceNum;
  unsigned int maxBlockSize;
  cudaGetDevice(&deviceNum);
  cudaDeviceGetAttribute((int*)&maxBlockSize, cudaDevAttrMaxThreadsPerBlock, deviceNum);
  maxBlockSize = floor(cbrt(maxBlockSize));


  dim3 block_size{maxBlockSize,maxBlockSize,maxBlockSize};
  dim3 grid_size(integer_divide_up(sizes->num_x,maxBlockSize), integer_divide_up(sizes->num_y,maxBlockSize), integer_divide_up(sizes->num_z, maxBlockSize));


  CUDA_SAFE_CALL(cudaMalloc(&oriData_d, sizeof(cl_float) * sizes->num_elements));
  CUDA_SAFE_CALL(cudaMalloc(&sizes_d, sizeof(sz_opencl_sizes)));
  CUDA_SAFE_CALL(cudaMalloc(&data_buffer_d, sizeof(cl_float) * sizes->num_blocks * sizes->max_num_block_elements));

  CUDA_SAFE_CALL(cudaMemcpy(oriData_d, oriData, sizeof(cl_float) * sizes->num_elements, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(sizes_d, sizes, sizeof(struct sz_opencl_sizes), cudaMemcpyHostToDevice));
  //do not copy data buffer since it is created in this function

  CUDA_SAFE_KERNEL_CALL((prepare_data_buffer_kernel<<<grid_size, block_size>>>(oriData_d, sizes_d, data_buffer_d)));

  CUDA_SAFE_CALL(cudaMemcpy(data_buffer, data_buffer_d, sizeof(cl_float) * sizes->num_blocks * sizes->max_num_block_elements, cudaMemcpyDeviceToHost));
  //do not copy sizes_d or oriData_d since they are const copied

  CUDA_SAFE_CALL(cudaFree(oriData_d));
  CUDA_SAFE_CALL(cudaFree(sizes_d));
  CUDA_SAFE_CALL(cudaFree(data_buffer_d));
}

__device__
void
compute_errors_kernel(const float* reg_params_pos, const float* data_buffer,
               sz_opencl_sizes const* sizes, float mean, float noise,
               bool use_mean, size_t i, size_t j, size_t k,
               float& err_sz, float& err_reg)
{
  const float* cur_data_pos =
      data_buffer +
          i * sizes->block_size * sizes->block_size +
          j * sizes->block_size + k;
  float curData = *cur_data_pos;
  float pred_sz =
      cur_data_pos[-1] + cur_data_pos[-sizes->strip_dim1_offset] +
          cur_data_pos[-sizes->strip_dim0_offset] -
          cur_data_pos[-sizes->strip_dim1_offset - 1] -
          cur_data_pos[-sizes->strip_dim0_offset - 1] -
          cur_data_pos[-sizes->strip_dim0_offset - sizes->strip_dim1_offset] +
          cur_data_pos[-sizes->strip_dim0_offset - sizes->strip_dim1_offset - 1];
  float pred_reg = reg_params_pos[0] * i +
      reg_params_pos[sizes->params_offset_b] * j +
      reg_params_pos[sizes->params_offset_c] * k +
      reg_params_pos[sizes->params_offset_d];
  if (use_mean) {
    err_sz += min(fabs(pred_sz - curData) + noise, fabs(mean - curData));
    err_reg += fabs(pred_reg - curData);
  } else {
    err_sz += fabs(pred_sz - curData) + noise;
    err_reg += fabs(pred_reg - curData);
  }
}


__global__
void
opencl_sample_kernel(const sz_opencl_sizes* sizes,
                     float mean,
                     float noise,
                     bool use_mean,
                     const float* data_buffer,
                     const float* reg_params_pos,
                     unsigned char* indicator_pos
) {
  unsigned long i = threadIdx.x + blockIdx.x * blockDim.x;//get_global_id(0);
  unsigned long j = threadIdx.y + blockIdx.y * blockDim.y;//get_global_id(1);
  unsigned long k = threadIdx.z + blockIdx.z * blockDim.z;//get_global_id(2);
  const unsigned int block_id = i * (sizes->num_y * sizes->num_z) + j * sizes->num_z + k;
  if(block_id < sizes->num_blocks) {
    const float *data_pos = data_buffer + (block_id * sizes->max_num_block_elements);
    /*sampling and decide which predictor*/
    {
      // sample point [1, 1, 1] [1, 1, 4] [1, 4, 1] [1, 4, 4] [4, 1, 1] [4,
      // 1, 4] [4, 4, 1] [4, 4, 4]
      float err_sz = 0.0, err_reg = 0.0;
      for (size_t block_i = 1; block_i < sizes->block_size; block_i++) {
        int bmi = sizes->block_size - block_i;
        compute_errors_kernel(&reg_params_pos[block_id], data_pos, sizes, mean, noise,
                              use_mean, block_i, block_i, block_i, err_sz,
                              err_reg);
        compute_errors_kernel(&reg_params_pos[block_id], data_pos, sizes, mean, noise,
                              use_mean, block_i, block_i, bmi, err_sz, err_reg);

        compute_errors_kernel(&reg_params_pos[block_id], data_pos, sizes, mean, noise,
                              use_mean, block_i, bmi, block_i, err_sz, err_reg);

        compute_errors_kernel(&reg_params_pos[block_id], data_pos, sizes, mean, noise,
                              use_mean, block_i, bmi, bmi, err_sz, err_reg);
      }
      indicator_pos[(i * sizes->num_y + j) * sizes->num_z + k] = err_reg >= err_sz;
    }
  }
}

void
opencl_sample_host(const sz_opencl_sizes* sizes,
              float mean,
              float noise,
              bool use_mean,
              const float* data_buffer,
              const float* reg_params_pos,
              unsigned char* indicator_pos
)
{
  int deviceNum;
  unsigned int maxBlockSize;
  cudaGetDevice(&deviceNum);
  cudaDeviceGetAttribute((int*)&maxBlockSize, cudaDevAttrMaxThreadsPerBlock, deviceNum);
  maxBlockSize = floor(cbrt(maxBlockSize));

  dim3 block_size{maxBlockSize,maxBlockSize,maxBlockSize};
  dim3 grid_size(integer_divide_up(sizes->num_x,maxBlockSize), integer_divide_up(sizes->num_y,maxBlockSize), integer_divide_up(sizes->num_z, maxBlockSize));

  sz_opencl_sizes* sizes_d;
  float* data_buffer_d;
  float* reg_params_pos_d;
  unsigned char* indicator_pos_d;

  CUDA_SAFE_CALL(cudaMalloc(&data_buffer_d, sizeof(cl_float) * sizes->data_buffer_size));
  CUDA_SAFE_CALL(cudaMalloc(&sizes_d, sizeof(sz_opencl_sizes)));
  CUDA_SAFE_CALL(cudaMalloc(&reg_params_pos_d, sizeof(cl_float) * sizes->reg_params_buffer_size));
  CUDA_SAFE_CALL(cudaMalloc(&indicator_pos_d, sizes->num_blocks * sizeof(unsigned char)));

  CUDA_SAFE_CALL(cudaMemcpy(data_buffer_d, data_buffer, sizeof(cl_float) * sizes->data_buffer_size, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(sizes_d, sizes, sizeof(sz_opencl_sizes), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(reg_params_pos_d, reg_params_pos, sizeof(cl_float) * sizes->reg_params_buffer_size, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(indicator_pos_d, indicator_pos, sizes->num_blocks * sizeof(unsigned char), cudaMemcpyHostToDevice));

  CUDA_SAFE_KERNEL_CALL((opencl_sample_kernel<<<grid_size,block_size>>>(sizes_d, mean, noise, use_mean, data_buffer_d, reg_params_pos_d, indicator_pos_d)));

  CUDA_SAFE_CALL(cudaMemcpy(indicator_pos, indicator_pos_d, sizes->num_blocks * sizeof(unsigned char), cudaMemcpyDeviceToHost));

  CUDA_SAFE_CALL(cudaFree(data_buffer_d));
  CUDA_SAFE_CALL(cudaFree(sizes_d));
  CUDA_SAFE_CALL(cudaFree(reg_params_pos_d));
  CUDA_SAFE_CALL(cudaFree(indicator_pos_d));


}
