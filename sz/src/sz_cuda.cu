/**
 *
 * A OpenCL/Cuda code for SZ
 * 
 * Developed by Robert Underwood while he was at Clemson University This
 * material is based upon work supported by the National Science Foundation
 * under Grant No. 1633608.
 *
 * This code is a derivative work of the SZ random access code by Sheng Di,
 * Franck Capello, et al at Argonne National Lab
 * 
 */
#include "cuda.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include "cuda_runtime_api.h"
#include "sz_opencl_kernels.h"

/**
  calls a CUDA api and returns an error if there was one
  */
#define CUDA_SAFE_CALL(call) {                                    \
  cudaError err = call;                                                    \
  if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
        __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                  \
  } \
}

/**
  calls a cuda kernel and returns an error if there was one
  */
#define CUDA_SAFE_KERNEL_CALL(call) {                                    \
	call; \
	cudaError_t err = cudaGetLastError(); \
  if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
        __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                  \
  } \
}

/**
  preforms integer division that rounds up instead of down

  throws a domain error if the division results in overflow
  */
template <class  T, class U>
auto integer_divide_up(T a, U b) {
  size_t val = (a % b != 0) ? (a/b+1) : (a/b);
  if(val > std::numeric_limits<decltype(a+b)>::max())
    throw std::domain_error("invalid integer division");
  else return val;
}

/**
  returns the max block size for a square kernel of a given dimension for the current cuda device

  Requires dimension be between 1 and 3 inclusive
  */
int max_block_size(int dim) {
  int deviceNum;
  unsigned int maxBlockSize;
  cudaGetDevice(&deviceNum);
  cudaDeviceGetAttribute((int*)&maxBlockSize, cudaDevAttrMaxThreadsPerBlock, deviceNum);
  switch(dim)
  {
    case 3:
      return floor(cbrt(maxBlockSize));
    case 2: 
      return floor(sqrt(maxBlockSize));
    case 1:
      return maxBlockSize;
    default:
      throw std::domain_error("invalid kernel dimension");
  }
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
  unsigned int maxBlockSize = max_block_size(3);

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

  unsigned int maxBlockSize2 = max_block_size(2);

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

  unsigned int maxBlockSize = max_block_size(3);

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
  unsigned int maxBlockSize = max_block_size(3);

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

__global__
void save_unpredictable_body_kernel(const sz_opencl_sizes *sizes,
                                    double realPrecision,
                                    float mean,
                                    bool use_mean,
                                    int intvRadius,
                                    int intvCapacity,
                                    int intvCapacity_sz,
                                    const float *reg_params,
                                    const unsigned char *indicator,
                                    const size_t *reg_params_pos_index,
                                    float *data_buffer,
                                    int *blockwise_unpred_count,
                                    float *unpredictable_data,
                                    int *result_type)
{
  unsigned long i = threadIdx.x + blockIdx.x * blockDim.x;//get_global_id(0);
  unsigned long j = threadIdx.y + blockIdx.y * blockDim.y;//get_global_id(1);
  unsigned long k = threadIdx.z + blockIdx.z * blockDim.z;//get_global_id(2);
  const size_t block_id = i * (sizes->num_y * sizes->num_z) + j * sizes->num_z + k;
  unsigned char indic = indicator[block_id];
  size_t block_data_pos = block_id*sizes->max_num_block_elements;
  float* data_pos = data_buffer + block_data_pos;
  size_t unpred_data_pos = block_data_pos;
  int *type = result_type + block_data_pos;
  if (!indic) {
    float curData;
    float pred;
    double itvNum;
    double diff;
    size_t index = 0;
    size_t block_unpredictable_count = 0;
    float* cur_data_pos = data_pos;
    //locate regression parameters' positions
    const float* reg_params_pos = reg_params + reg_params_pos_index[block_id];

    for (size_t ii = 0; ii < sizes->block_size; ii++) {
      for (size_t jj = 0; jj < sizes->block_size; jj++) {
        for (size_t kk = 0; kk < sizes->block_size; kk++) {
          curData = *cur_data_pos;
          pred = reg_params_pos[0] * ii +
              reg_params_pos[sizes->params_offset_b] * jj +
              reg_params_pos[sizes->params_offset_c] * kk +
              reg_params_pos[sizes->params_offset_d];
          diff = curData - pred;
          itvNum = fabs(diff) / realPrecision + 1;
          if (itvNum < intvCapacity) {
            if (diff < 0)
              itvNum = -itvNum;
            type[index] = (int)(itvNum / 2) + intvRadius;
            pred = pred + 2 * (type[index] - intvRadius) * realPrecision;
            // ganrantee comporession error against the case of
            // machine-epsilon
            if (fabs(curData - pred) > realPrecision) {
              type[index] = 0;
              unpredictable_data[unpred_data_pos+block_unpredictable_count++] = curData;
            }
          } else {
            type[index] = 0;
            unpredictable_data[unpred_data_pos+block_unpredictable_count++] = curData;
          }
          index++;
          cur_data_pos++;
        }
      }
    }
    blockwise_unpred_count[block_id] = block_unpredictable_count;
  } else {
    // use SZ
    // SZ predication
    float* cur_data_pos = data_pos;
    float curData;
    float pred3D;
    double itvNum, diff;
    size_t index = 0;
    size_t block_unpredictable_count = 0;
    for (size_t ii = 0; ii < sizes->block_size; ii++) {
      for (size_t jj = 0; jj < sizes->block_size; jj++) {
        for (size_t kk = 0; kk < sizes->block_size; kk++) {
          curData = *cur_data_pos;
          if (use_mean && fabs(curData - mean) <= realPrecision) {
            type[index] = 1;
            *cur_data_pos = mean;
          } else {
            float d000, d001, d010, d011, d100, d101, d110;
            d000 = d001 = d010 = d011 = d100 = d101 = d110 = 1;
            if(ii == 0){
              d000 = d001 = d010 = d011 = 0;
            }
            if(jj == 0){
              d000 = d001 = d100 = d101 = 0;
            }
            if(kk == 0){
              d000 = d010 = d100 = d110 = 0;
            }
            d000 = d000 ? cur_data_pos[-sizes->strip_dim0_offset - sizes->strip_dim1_offset - 1] : 0;
            d001 = d001 ? cur_data_pos[-sizes->strip_dim0_offset - sizes->strip_dim1_offset] : 0;
            d010 = d010 ? cur_data_pos[-sizes->strip_dim0_offset - 1] : 0;
            d011 = d011 ? cur_data_pos[-sizes->strip_dim0_offset] : 0;
            d100 = d100 ? cur_data_pos[-sizes->strip_dim1_offset - 1] : 0;
            d101 = d101 ? cur_data_pos[-sizes->strip_dim1_offset] : 0;
            d110 = d110 ? cur_data_pos[- 1] : 0;

            pred3D = d110 + d101 + d011 - d100 - d010 - d001 + d000;
            diff = curData - pred3D;
            itvNum = fabs(diff) / realPrecision + 1;
            if (itvNum < intvCapacity_sz) {
              if (diff < 0)
                itvNum = -itvNum;
              type[index] = (int)(itvNum / 2) + intvRadius;
              *cur_data_pos =
                  pred3D + 2 * (type[index] - intvRadius) * realPrecision;
              // ganrantee comporession error against the case of
              // machine-epsilon
              if (fabs(curData - *cur_data_pos) > realPrecision) {
                type[index] = 0;
                *cur_data_pos = curData;
                unpredictable_data[unpred_data_pos+block_unpredictable_count++] = curData;
              }
            } else {
              type[index] = 0;
              *cur_data_pos = curData;
              unpredictable_data[unpred_data_pos+block_unpredictable_count++] = curData;
            }
          }
          index++;
          cur_data_pos++;
        }
      }
    }
    blockwise_unpred_count[block_id] = block_unpredictable_count;
  } // end SZ
}

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
                                  int *result_type) {
  sz_opencl_sizes *sizes_d;
  float *data_buffer_d;
  float *unpredictable_data_d;
  int *result_type_d;
  float *reg_params_d;
  unsigned char *indicator_d;
  int *blockwise_unpred_count_d;
  unsigned long *reg_params_pos_index_d;

  unsigned int maxBlockSize = max_block_size(3);
  dim3 block_size(maxBlockSize, maxBlockSize, maxBlockSize);
  dim3 grid_size(integer_divide_up(sizes->num_x,maxBlockSize), integer_divide_up(sizes->num_y,maxBlockSize), integer_divide_up(sizes->num_z,maxBlockSize));

  CUDA_SAFE_CALL(cudaMalloc(&sizes_d, sizeof(sz_opencl_sizes))); 
  CUDA_SAFE_CALL(cudaMalloc(&data_buffer_d, sizeof(float) * sizes->data_buffer_size)); 
  CUDA_SAFE_CALL(cudaMalloc(&unpredictable_data_d, sizeof(float) * sizes->unpred_data_max_size * sizes->block_size)); 
  CUDA_SAFE_CALL(cudaMalloc(&result_type_d, sizeof(int) * sizes->data_buffer_size)); 
  CUDA_SAFE_CALL(cudaMalloc(&reg_params_d, sizeof(float)* sizes->reg_params_buffer_size)); 
  CUDA_SAFE_CALL(cudaMalloc(&indicator_d, sizeof(unsigned char) * sizes->num_blocks)); 
  CUDA_SAFE_CALL(cudaMalloc(&blockwise_unpred_count_d, sizeof(int) * sizes->num_blocks)); 
  CUDA_SAFE_CALL(cudaMalloc(&reg_params_pos_index_d, sizeof(unsigned long) * sizes->reg_params_buffer_size));

  CUDA_SAFE_CALL(cudaMemcpy(sizes_d,sizes, sizeof(sz_opencl_sizes), cudaMemcpyHostToDevice)); 
  CUDA_SAFE_CALL(cudaMemcpy(data_buffer_d,data_buffer, sizeof(float) * sizes->data_buffer_size, cudaMemcpyHostToDevice)); 
  CUDA_SAFE_CALL(cudaMemcpy(unpredictable_data_d,unpredictable_data, sizeof(float) * sizes->unpred_data_max_size * sizes->block_size, cudaMemcpyHostToDevice)); 
  CUDA_SAFE_CALL(cudaMemcpy(result_type_d,result_type, sizeof(int) * sizes->data_buffer_size, cudaMemcpyHostToDevice)); 
  CUDA_SAFE_CALL(cudaMemcpy(reg_params_d,reg_params, sizeof(float)* sizes->reg_params_buffer_size, cudaMemcpyHostToDevice)); 
  CUDA_SAFE_CALL(cudaMemcpy(indicator_d,indicator, sizeof(unsigned char) * sizes->num_blocks, cudaMemcpyHostToDevice)); 
  CUDA_SAFE_CALL(cudaMemcpy(blockwise_unpred_count_d,blockwise_unpred_count, sizeof(int) * sizes->num_blocks, cudaMemcpyHostToDevice)); 
  CUDA_SAFE_CALL(cudaMemcpy(reg_params_pos_index_d,reg_params_pos_index, sizeof(unsigned long) * sizes->reg_params_buffer_size, cudaMemcpyHostToDevice));

  save_unpredictable_body_kernel <<<grid_size,block_size>>>(sizes_d,
                                 realPrecision,
                                 mean,
                                 use_mean,
                                 intvRadius,
                                 intvCapacity,
                                 intvCapacity_sz,
                                 reg_params_d,
                                 indicator_d,
                                 reg_params_pos_index_d,
                                 data_buffer_d,
                                 blockwise_unpred_count_d,
                                 unpredictable_data_d,
                                 result_type
                                 );

  //donot copy reg_params, indicator, or reg_params_pos_index back since they are const
  CUDA_SAFE_CALL(cudaMemcpy(data_buffer,data_buffer_d, sizeof(float) * sizes->data_buffer_size, cudaMemcpyDeviceToHost)); 
  CUDA_SAFE_CALL(cudaMemcpy(unpredictable_data,unpredictable_data_d, sizeof(float) * sizes->unpred_data_max_size * sizes->block_size, cudaMemcpyDeviceToHost)); 
  CUDA_SAFE_CALL(cudaMemcpy(result_type,result_type_d, sizeof(int) * sizes->data_buffer_size, cudaMemcpyDeviceToHost)); 
  CUDA_SAFE_CALL(cudaMemcpy(blockwise_unpred_count,blockwise_unpred_count_d, sizeof(int) * sizes->num_blocks, cudaMemcpyDeviceToHost));


  CUDA_SAFE_CALL(cudaFree(sizes_d));
  CUDA_SAFE_CALL(cudaFree(data_buffer_d));
  CUDA_SAFE_CALL(cudaFree(unpredictable_data_d));
  CUDA_SAFE_CALL(cudaFree(result_type_d));
  CUDA_SAFE_CALL(cudaFree(reg_params_d));
  CUDA_SAFE_CALL(cudaFree(indicator_d));
  CUDA_SAFE_CALL(cudaFree(blockwise_unpred_count_d));
  CUDA_SAFE_CALL(cudaFree(reg_params_pos_index_d));


}


__device__
void
decompress_location_using_regression_kernel(const sz_opencl_sizes* sizes, const sz_opencl_decompress_positions* pos,
                                     const float* reg_params_pos,
                                     const int* type, const float* block_unpred,
                                     double realPrecision, int intvRadius,
                                     float* data_out)
{
  size_t unpredictable_count = 0;
  //TODO refactor unpredictable count in to a pre-scan to remove dependance on unpredictable_count
  //#pragma omp parallel for collapse(3)
  for (size_t ii = 0; ii < sizes->block_size; ii++) {
    for (size_t jj = 0; jj < sizes->block_size; jj++) {
      for (size_t kk = 0; kk < sizes->block_size; kk++) {
        size_t index = (ii*sizes->block_size*sizes->block_size) + (jj*sizes->block_size) + kk;
        int type_ = type[index];
        if (type_ != 0) {
          float pred = reg_params_pos[0] * ii + reg_params_pos[1] * jj +
              reg_params_pos[2] * kk + reg_params_pos[3];
          data_out[ii * pos->dec_block_dim0_offset + jj * pos->dec_block_dim1_offset +
              kk] = pred + 2 * (type_ - intvRadius) * realPrecision;
        } else {
          data_out[ii * pos->dec_block_dim0_offset + jj * pos->dec_block_dim1_offset +
              kk] = block_unpred[unpredictable_count++];
        }
      }
    }
  }
}

__device__
void
decompress_location_using_sz_kernel(const sz_opencl_sizes* sizes, const sz_opencl_decompress_positions* pos, double realPrecision,
                             float mean, unsigned char use_mean, int intvRadius,
                             const int* type, float* data_pos,
                             const float* block_unpred)
{
  float* cur_data_pos;
  float pred;
  size_t index = 0;
  int type_;
  size_t unpredictable_count = 0;
  //DO NOT parallelize this loop too small to matter!
  for (size_t ii = 0; ii < sizes->block_size; ii++) {
    for (size_t jj = 0; jj < sizes->block_size; jj++) {
      for (size_t kk = 0; kk < sizes->block_size; kk++) {
        cur_data_pos = data_pos + ii * pos->dec_block_dim0_offset +
                         jj * pos->dec_block_dim1_offset + kk;
        type_ = type[index];
        if (use_mean && type_ == 1) {
          *cur_data_pos = mean;
        } else if (type_ == 0) {
          *cur_data_pos = block_unpred[unpredictable_count++];
        } else {
          float d000, d001, d010, d011, d100, d101, d110;
          d000 = d001 = d010 = d011 = d100 = d101 = d110 = 1;
          if(ii == 0){
            d000 = d001 = d010 = d011 = 0;
          }
          if(jj == 0){
            d000 = d001 = d100 = d101 = 0;
          }
          if(kk == 0){
            d000 = d010 = d100 = d110 = 0;
          }
          d000 = d000 ? cur_data_pos[-pos->dec_block_dim0_offset - pos->dec_block_dim1_offset - 1] : 0;
          d001 = d001 ? cur_data_pos[-pos->dec_block_dim0_offset - pos->dec_block_dim1_offset] : 0;
          d010 = d010 ? cur_data_pos[-pos->dec_block_dim0_offset - 1] : 0;
          d011 = d011 ? cur_data_pos[-pos->dec_block_dim0_offset] : 0;
          d100 = d100 ? cur_data_pos[-pos->dec_block_dim1_offset - 1] : 0;
          d101 = d101 ? cur_data_pos[-pos->dec_block_dim1_offset] : 0;
          d110 = d110 ? cur_data_pos[- 1] : 0;
          // pred =
          //   block_data_pos[-1] + block_data_pos[-sizes->block_dim1_offset] +
          //   block_data_pos[-sizes->block_dim0_offset] -
          //   block_data_pos[-sizes->block_dim1_offset - 1] -
          //   block_data_pos[-sizes->block_dim0_offset - 1] -
          //   block_data_pos[-sizes->block_dim0_offset - sizes->block_dim1_offset] +
          //   block_data_pos[-sizes->block_dim0_offset - sizes->block_dim1_offset -
          //                  1];
          pred = d110 + d101 + d011 - d100 - d010 - d001 + d000;
          *cur_data_pos = pred + 2 * (type_ - intvRadius) * realPrecision;
        }
        index++;
      }
    }
  }
}

__global__
void decompress_all_blocks_kernel(const sz_opencl_sizes* sizes,
                                double realPrecision, float mean, unsigned char use_mean,
                                const unsigned char* indicator, const float* reg_params,
                                int intvRadius, const size_t* unpred_offset,
                                const float* unpred_data,
                                const sz_opencl_decompress_positions* pos,
                                const int* result_type,
                                float* dec_block_data)
{
  unsigned long i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long j = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned long k = threadIdx.z + blockIdx.z * blockDim.z;
  float* data_pos = dec_block_data + (i - pos->start_block1)*sizes->block_size*pos->dec_block_dim0_offset + (j - pos->start_block2)*sizes->block_size*pos->dec_block_dim1_offset + (k - pos->start_block3)*sizes->block_size;
  const int* type = result_type +
      (i - pos->start_block1) * sizes->block_size * sizes->block_size *
          (pos->num_data_blocks2) * sizes->block_size *
          (pos->num_data_blocks3) +
      (j - pos->start_block2) * sizes->max_num_block_elements *
          (pos->num_data_blocks3) +
      (k - pos->start_block3) * sizes->max_num_block_elements;
  size_t coeff_index =
      i * sizes->num_y * sizes->num_z + j * sizes->num_z + k;
  const float* block_unpred = unpred_data + unpred_offset[coeff_index];
  if (indicator[coeff_index]) {
    // decompress by SZ
    decompress_location_using_sz_kernel(sizes, pos, realPrecision, mean, use_mean,
                                 intvRadius, type, data_pos,
                                 block_unpred);
  } else {
    // decompress by regression
    decompress_location_using_regression_kernel(
        sizes, pos, reg_params + 4 * coeff_index, type, block_unpred,
        realPrecision, intvRadius, data_pos);
  }
}

void decompress_all_blocks_host(const sz_opencl_sizes* sizes,
                           double realPrecision, float mean, unsigned char use_mean,
                           const unsigned char* indicator, const float* reg_params,
                           int intvRadius, const size_t* unpred_offset,
                           const float* unpred_data,
                           const sz_opencl_decompress_positions* pos,
                           const int* result_type,
                           float* dec_block_data, size_t data_unpred_size)
{
sz_opencl_sizes* sizes_d;
unsigned char* indicator_d;
float* reg_params_d;
size_t* unpred_offset_d;
float* unpred_data_d;
sz_opencl_decompress_positions* pos_d;
int* result_type_d;
float* dec_block_data_d;

unsigned int maxBlockSize = max_block_size(3);
dim3 block_size(maxBlockSize, maxBlockSize, maxBlockSize);
dim3 grid_size(integer_divide_up(pos->num_data_blocks1,maxBlockSize), integer_divide_up(pos->num_data_blocks2,maxBlockSize), integer_divide_up(pos->num_data_blocks2,maxBlockSize));


CUDA_SAFE_CALL(cudaMalloc(&sizes_d, sizeof(sz_opencl_sizes)));
CUDA_SAFE_CALL(cudaMalloc(&indicator_d, sizeof(unsigned char) * sizes->num_blocks));
CUDA_SAFE_CALL(cudaMalloc(&reg_params_d, sizeof(float)* sizes->reg_params_buffer_size));
CUDA_SAFE_CALL(cudaMalloc(&unpred_offset_d, sizeof(size_t) * sizes->num_blocks));
CUDA_SAFE_CALL(cudaMalloc(&unpred_data_d, sizeof(float)* data_unpred_size));
CUDA_SAFE_CALL(cudaMalloc(&pos_d, sizeof(sz_opencl_decompress_positions)));
CUDA_SAFE_CALL(cudaMalloc(&result_type_d, sizeof(int)* sizes->data_buffer_size));
CUDA_SAFE_CALL(cudaMalloc(&dec_block_data_d, sizeof(float) * pos->dec_block_data_size));


CUDA_SAFE_CALL(cudaMemcpy(sizes_d,sizes, sizeof(sz_opencl_sizes), cudaMemcpyHostToDevice));
CUDA_SAFE_CALL(cudaMemcpy(indicator_d,indicator, sizeof(unsigned char) * sizes->num_blocks, cudaMemcpyHostToDevice));
CUDA_SAFE_CALL(cudaMemcpy(reg_params_d,reg_params, sizeof(float)* sizes->reg_params_buffer_size, cudaMemcpyHostToDevice));
CUDA_SAFE_CALL(cudaMemcpy(unpred_offset_d,unpred_offset, sizeof(size_t) * sizes->num_blocks, cudaMemcpyHostToDevice));
CUDA_SAFE_CALL(cudaMemcpy(unpred_data_d,unpred_data, sizeof(float)* data_unpred_size, cudaMemcpyHostToDevice));
CUDA_SAFE_CALL(cudaMemcpy(pos_d,pos, sizeof(sz_opencl_decompress_positions), cudaMemcpyHostToDevice));
CUDA_SAFE_CALL(cudaMemcpy(result_type_d,result_type, sizeof(int)* sizes->data_buffer_size, cudaMemcpyHostToDevice));
CUDA_SAFE_CALL(cudaMemcpy(dec_block_data_d,dec_block_data, sizeof(float) * pos->dec_block_data_size, cudaMemcpyHostToDevice));

decompress_all_blocks_kernel<<<grid_size, block_size>>>(sizes_d,
                                realPrecision, mean, use_mean,
                                indicator_d, reg_params_d,
                                intvRadius, unpred_offset_d,
                                unpred_data_d,
                                pos_d,
                                result_type_d,
                                dec_block_data_d);


//only dec_block_data is non-const, so only copy it back
CUDA_SAFE_CALL(cudaMemcpy(dec_block_data,dec_block_data_d, sizeof(float) * pos->dec_block_data_size, cudaMemcpyDeviceToHost));


CUDA_SAFE_CALL(cudaFree(sizes_d));
CUDA_SAFE_CALL(cudaFree(indicator_d));
CUDA_SAFE_CALL(cudaFree(reg_params_d));
CUDA_SAFE_CALL(cudaFree(unpred_offset_d));
CUDA_SAFE_CALL(cudaFree(unpred_data_d));
CUDA_SAFE_CALL(cudaFree(pos_d));
CUDA_SAFE_CALL(cudaFree(result_type_d));
CUDA_SAFE_CALL(cudaFree(dec_block_data_d));


}
