#include "sz_opencl.h"
#include "sz_cuda.h"
#include "sz.h"
#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sz_opencl_config.h"

#if !SZ_OPENCL_USE_CUDA
#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"
#endif

#include "sz_opencl_host_utils.h"
#include "sz_opencl_kernels.h"
#include "sz_opencl_private.h"

float
compute_mean(const float* oriData, double realPrecision, float dense_pos,
             size_t num_elements)
{
  float mean = 0.0f;
  {
    // compute mean
    double sum = 0.0;
    size_t mean_count = 0;
    for (size_t i = 0; i < num_elements; i++) {
      if (fabs(oriData[i] - dense_pos) < realPrecision) {
        sum += oriData[i];
        mean_count++;
      }
    }
    if (mean_count > 0)
      mean = sum / mean_count;
  }
  return mean;
}

unsigned char*
encode_all_blocks(sz_opencl_sizes const* sizes, int* result_type, int* type,
                  HuffmanTree* huffmanTree, unsigned char* result_pos)
{
  type = result_type;
  size_t total_type_array_size = 0;
  unsigned char* type_array_buffer = (unsigned char*)malloc(
    sizes->num_blocks * sizes->max_num_block_elements * sizeof(int));
  unsigned short* type_array_block_size =
    (unsigned short*)malloc(sizes->num_blocks * sizeof(unsigned short));
  
  size_t num_yz = sizes->num_y*sizes->num_z;
  //don't parallelize this loop with GPU, encode function doesn't preform well on GPU
  //use new encoders with OpenCL/CUDA when available
  #pragma omp parallel for collapse(3)
  for (size_t i = 0; i < sizes->num_x; i++) {
    for (size_t j = 0; j < sizes->num_y; j++) {
      for (size_t k = 0; k < sizes->num_z; k++) {
		size_t block_pos = (i*num_yz+j*sizes->num_z+k);
		size_t buff_pos = block_pos*sizes->max_num_block_elements;
		
        size_t typeArray_size = 0;
        encode(huffmanTree, &type[buff_pos], sizes->max_num_block_elements,
               &type_array_buffer[buff_pos], &typeArray_size);
        type_array_block_size[block_pos] = typeArray_size;
      }
    }
  } 
  
  size_t first_block_size = type_array_block_size[0];
  total_type_array_size += first_block_size;
  unsigned char* type_array_buffer_src_pos = type_array_buffer + sizes->max_num_block_elements;
  unsigned char* type_array_buffer_tgt_pos = type_array_buffer + first_block_size; //get the current position for compressed type array in the buffer
  for (size_t i = 1; i < sizes->num_blocks; i++)
  {
	 size_t cmpr_size = type_array_block_size[i];
	 total_type_array_size += cmpr_size;
	 memmove(type_array_buffer_tgt_pos, type_array_buffer_src_pos, cmpr_size);
	 type_array_buffer_src_pos += sizes->max_num_block_elements;
	 type_array_buffer_tgt_pos += cmpr_size;
  }
  
  size_t compressed_type_array_block_size;
  unsigned char* compressed_type_array_block = SZ_compress_args(
    SZ_UINT16, type_array_block_size, &compressed_type_array_block_size, ABS,
    0.5, 0, 0, 0, 0, 0, 0, sizes->num_blocks);
  memcpy(result_pos, &compressed_type_array_block_size, sizeof(size_t));
  result_pos += sizeof(size_t);
  memcpy(result_pos, compressed_type_array_block,
         compressed_type_array_block_size);
  result_pos += compressed_type_array_block_size;

  memcpy(result_pos, type_array_buffer, total_type_array_size);
  result_pos += total_type_array_size;

  free(compressed_type_array_block);
  free(type_array_buffer);
  free(type_array_block_size);
  return result_pos;
}

void
calculate_regression_coefficents(struct sz_opencl_state* state,
                                 const cl_float* oriData,
                                 sz_opencl_sizes const* sizes,
                                 cl_float* reg_params,
                                 cl_float* const pred_buffer)
{
#if SZ_OPENCL_USE_CUDA
	std::vector<float> oriData_cuda (oriData, oriData + sizes->num_elements);
	std::vector<float> reg_params_cuda (reg_params, reg_params + sizes->reg_params_buffer_size);
	std::vector<float> pred_buffer_cuda (pred_buffer, pred_buffer + (sizes->num_blocks * sizes->max_num_block_elements));

  calculate_regression_coefficents_host(oriData_cuda.data(), sizes, reg_params_cuda.data(), pred_buffer_cuda.data());  
#endif

#pragma omp parallel for collapse(3)
for (cl_ulong i = 0; i < sizes->num_x; i++) {
  for (cl_ulong j = 0; j < sizes->num_y; j++) {
    for (cl_ulong k = 0; k < sizes->num_z; k++) {

      const unsigned int block_id =
        i * (sizes->num_y * sizes->num_z) + j * sizes->num_z + k;
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
}

}

void save_unpredictable_body(const sz_opencl_sizes *sizes,
                             double realPrecision,
                             float mean,
                             bool use_mean,
                             float *data_buffer,
                             const float *reg_params,
                             const unsigned char *indicator,
                             const size_t *reg_params_pos_index,
                             int *blockwise_unpred_count,
                             float *unpredictable_data,
                             int *result_type) {
  int *type;
  int intvRadius = exe_params->intvRadius;
  int intvCapacity = exe_params->intvCapacity;
  int intvCapacity_sz = exe_params->intvCapacity - 2;
  type = result_type;

#if SZ_OPENCL_USE_CUDA
  //this kernel requires too much memory on the GPU to port now

  //save_unpredictable_body_host(sizes,
  //                             realPrecision,
  //                             mean,
  //                             use_mean,
  //                             intvRadius,
  //                             intvCapacity,
  //                             intvCapacity_sz,
  //                             reg_params,
  //                             indicator,
  //                             reg_params_pos_index,
  //                             data_buffer,
  //                             blockwise_unpred_count,
  //                             unpredictable_data,
  //                             result_type);
#endif

  //TODO parallelize this loop
#pragma omp parallel for collapse(3)
  for (size_t i = 0; i < sizes->num_x; i++) {
    for (size_t j = 0; j < sizes->num_y; j++) {
      for (size_t k = 0; k < sizes->num_z; k++) {
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

                  // pred3D = cur_data_pos[-1] +
                  //          cur_data_pos[-sizes->strip_dim1_offset] +
                  //          cur_data_pos[-sizes->strip_dim0_offset] -
                  //          cur_data_pos[-sizes->strip_dim1_offset - 1] -
                  //          cur_data_pos[-sizes->strip_dim0_offset - 1] -
                  //          cur_data_pos[-sizes->strip_dim0_offset -
                  //                       sizes->strip_dim1_offset] +
                  //          cur_data_pos[-sizes->strip_dim0_offset -
                  //                       sizes->strip_dim1_offset - 1];
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
      } // end k
    } // end j
  }   // end i

}
size_t
save_unpredictable_data(sz_opencl_sizes const *sizes,
                        double realPrecision,
                        float mean,
                        bool use_mean,
                        float *data_buffer,
                        float *unpredictable_data,
                        int *result_type,
                        float *reg_params,
                        const unsigned char *indicator,
                        int *blockwise_unpred_count)
{
  size_t total_unpred = 0;	
  //Pre-scanning indicator to get the reg_params_pos of regression params for each block id {i,j,k}
  size_t* reg_params_pos_index = (size_t*)malloc(sizes->num_blocks*sizeof(size_t));
  memset(reg_params_pos_index, 0, sizes->num_blocks*sizeof(size_t));
  size_t counter = 0;
  for (size_t i = 0; i < sizes->num_x; i++) {
    for (size_t j = 0; j < sizes->num_y; j++) {
      for (size_t k = 0; k < sizes->num_z; k++) {
		   size_t block_id = i*sizes->num_y*sizes->num_z + j*sizes->num_z + k;
		   unsigned char indic = indicator[block_id];
		   if(!indic)
			   reg_params_pos_index[block_id] = counter++;
	  }
    }
  }

  save_unpredictable_body(sizes,
                          realPrecision,
                          mean,
                          use_mean,
                          data_buffer,
                          reg_params,
                          indicator,
                          reg_params_pos_index,
                          blockwise_unpred_count,
                          unpredictable_data,
                          result_type);

  free(reg_params_pos_index);

  //TODO: compute the total_unpred;
  size_t first_unpred_count = blockwise_unpred_count[0];
  total_unpred += first_unpred_count;
  float* src_pos = unpredictable_data + sizes->max_num_block_elements;
  float* tgt_pos = unpredictable_data + first_unpred_count; //get the current position for compressed type array in the buffer
  for(size_t i = 1;i < sizes->num_blocks;i++)
  {
	  size_t unpred_count = blockwise_unpred_count[i];
	  total_unpred += unpred_count;
	  memmove(tgt_pos, src_pos, unpred_count*sizeof(float));
	  src_pos += sizes->max_num_block_elements;
	  tgt_pos += unpred_count;
  } 
  
  return total_unpred;
}

void
compute_errors(const float* reg_params_pos, const float* data_buffer,
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
    err_sz += std::min(fabs(pred_sz - curData) + noise, fabs(mean - curData));
    err_reg += fabs(pred_reg - curData);
  } else {
    err_sz += fabs(pred_sz - curData) + noise;
    err_reg += fabs(pred_reg - curData);
  }
}

void
opencl_sample(const sz_opencl_sizes* sizes,
    float mean,
    float noise,
    bool use_mean,
    const float* data_buffer,
    float* reg_params_pos,
    unsigned char* indicator_pos
    )
{
#if SZ_OPENCL_USE_CUDA
  opencl_sample_host(sizes, mean, noise, use_mean, data_buffer, reg_params_pos, indicator_pos);
#else

//TODO refactor pred_buffer  to be loop independent
#pragma omp parallel for collapse(3) 
  for (size_t i = 0; i < sizes->num_x; i++) {
    for (size_t j = 0; j < sizes->num_y; j++) {
      for (size_t k = 0; k < sizes->num_z; k++) {
        const unsigned int block_id = i * (sizes->num_y * sizes->num_z) + j * sizes->num_z + k;
        const float * data_pos = data_buffer + (block_id * sizes->max_num_block_elements);
        /*sampling and decide which predictor*/
        {
          // sample point [1, 1, 1] [1, 1, 4] [1, 4, 1] [1, 4, 4] [4, 1, 1] [4,
          // 1, 4] [4, 4, 1] [4, 4, 4]
          float err_sz = 0.0, err_reg = 0.0;
          for (size_t block_i = 1; block_i < sizes->block_size; block_i++) {
            int bmi = sizes->block_size - block_i;
            compute_errors(&reg_params_pos[block_id], data_pos, sizes, mean, noise,
                           use_mean, block_i, block_i, block_i, err_sz,
                           err_reg);
            compute_errors(&reg_params_pos[block_id], data_pos, sizes, mean, noise,
                           use_mean, block_i, block_i, bmi, err_sz, err_reg);

            compute_errors(&reg_params_pos[block_id], data_pos, sizes, mean, noise,
                           use_mean, block_i, bmi, block_i, err_sz, err_reg);

            compute_errors(&reg_params_pos[block_id], data_pos, sizes, mean, noise,
                           use_mean, block_i, bmi, bmi, err_sz, err_reg);
          }
          indicator_pos[(i*sizes->num_y+j)*sizes->num_z+k] = err_reg >= err_sz;
        }
      }// end k
    }// end j
  }// end i
#endif
}

void
decompress_location_using_regression(const sz_opencl_sizes& sizes, const sz_opencl_decompress_positions& pos,
                                     const float* reg_params_pos,
                                     const int* type, const float* block_unpred,
                                     double realPrecision, int intvRadius,
                                     float* data_out)
{
  size_t unpredictable_count = 0;
  //TODO refactor unpredictable count in to a pre-scan to remove dependance on unpredictable_count
  //#pragma omp parallel for collapse(3)
  for (size_t ii = 0; ii < sizes.block_size; ii++) {
    for (size_t jj = 0; jj < sizes.block_size; jj++) {
      for (size_t kk = 0; kk < sizes.block_size; kk++) {
        size_t index = (ii*sizes.block_size*sizes.block_size) + (jj*sizes.block_size) + kk;
        int type_ = type[index];
        if (type_ != 0) {
          float pred = reg_params_pos[0] * ii + reg_params_pos[1] * jj +
                 reg_params_pos[2] * kk + reg_params_pos[3];
          data_out[ii * pos.dec_block_dim0_offset + jj * pos.dec_block_dim1_offset +
                   kk] = pred + 2 * (type_ - intvRadius) * realPrecision;
        } else {
          data_out[ii * pos.dec_block_dim0_offset + jj * pos.dec_block_dim1_offset +
                   kk] = block_unpred[unpredictable_count++];
        }
      }
    }
  }
}

void
decompress_location_using_sz(const sz_opencl_sizes& sizes, const sz_opencl_decompress_positions& pos, double realPrecision,
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
  for (size_t ii = 0; ii < sizes.block_size; ii++) {
    for (size_t jj = 0; jj < sizes.block_size; jj++) {
      for (size_t kk = 0; kk < sizes.block_size; kk++) {
        cur_data_pos = data_pos + ii * pos.dec_block_dim0_offset +
                         jj * pos.dec_block_dim1_offset + kk;
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
          d000 = d000 ? cur_data_pos[-pos.dec_block_dim0_offset - pos.dec_block_dim1_offset - 1] : 0;
          d001 = d001 ? cur_data_pos[-pos.dec_block_dim0_offset - pos.dec_block_dim1_offset] : 0;
          d010 = d010 ? cur_data_pos[-pos.dec_block_dim0_offset - 1] : 0;
          d011 = d011 ? cur_data_pos[-pos.dec_block_dim0_offset] : 0;
          d100 = d100 ? cur_data_pos[-pos.dec_block_dim1_offset - 1] : 0;
          d101 = d101 ? cur_data_pos[-pos.dec_block_dim1_offset] : 0;
          d110 = d110 ? cur_data_pos[- 1] : 0;          
          // pred =
          //   block_data_pos[-1] + block_data_pos[-sizes.block_dim1_offset] +
          //   block_data_pos[-sizes.block_dim0_offset] -
          //   block_data_pos[-sizes.block_dim1_offset - 1] -
          //   block_data_pos[-sizes.block_dim0_offset - 1] -
          //   block_data_pos[-sizes.block_dim0_offset - sizes.block_dim1_offset] +
          //   block_data_pos[-sizes.block_dim0_offset - sizes.block_dim1_offset -
          //                  1];
          pred = d110 + d101 + d011 - d100 - d010 - d001 + d000;
          *cur_data_pos = pred + 2 * (type_ - intvRadius) * realPrecision;
        }
        index++;
      }
    }
  }
}

void
decode_all_blocks(unsigned char* comp_data_pos, const sz_opencl_sizes& sizes,
                  node_t* root, const sz_opencl_decompress_positions& pos,
                  const size_t* type_array_offset, int* result_type)
{
  //not porting to GPU because this decoder is inefficent on GPU and newer decoders
  //are underdevelopment
  #pragma omp parallel for collapse(3)
  for (size_t i = pos.start_block1; i < pos.end_block1; i++) {
    for (size_t j = pos.start_block2; j < pos.end_block2; j++) {
      for (size_t k = pos.start_block3; k < pos.end_block3; k++) {
        size_t index = i * sizes.num_y * sizes.num_z + j * sizes.num_z + k;
	int* block_type = result_type + index * sizes.max_num_block_elements;
        decode(comp_data_pos + type_array_offset[index],
               sizes.max_num_block_elements, root, block_type);
      }
    }
  }
}
void
decompress_coefficents(const sz_opencl_sizes& sizes,
                       const unsigned char* indicator,
                       const int* coeff_intvRadius, int* const* coeff_type,
                       const double* precision, float* const* coeff_unpred_data,
                       float* last_coefficients, int* coeff_unpred_data_count,
                       float* reg_params)
{
  float* reg_params_pos = reg_params;
  size_t coeff_index = 0;

  //TODO coeff_index depends on the values in the indicator array; prescan
  //TODO reg_params_pos depends on indicator array and i
  //No need to parallel in our opinion, or can be done later - Xin and Sheng
  //#pragma omp parallel for
  for (size_t i = 0; i < sizes.num_blocks; i++) {
    if (!indicator[i]) {
      float pred;
      int type_;
      for (int e = 0; e < 4; e++) {
        type_ = coeff_type[e][coeff_index];
        if (type_ != 0) {
          pred = last_coefficients[e];
          last_coefficients[e] =
            pred + 2 * (type_ - coeff_intvRadius[e]) * precision[e];
        } else {
          last_coefficients[e] =
            coeff_unpred_data[e][coeff_unpred_data_count[e]];
          coeff_unpred_data_count[e]++;
        }
        reg_params_pos[e] = last_coefficients[e];
      }
      coeff_index++;
    }
    reg_params_pos += 4;
  }
}
sz_opencl_coefficient_params&
compress_coefficent_arrays(size_t reg_count,
                           const sz_opencl_coefficient_sizes& coefficient_sizes,
                           sz_opencl_coefficient_params& params)
{

  //TODO unpredictable count is not such that it can be parallelized
  /*
  //TODO tune the parallel threshold
  const int parallel_threshold = 100;
  #pragma omp parallel for if(reg_count > parallel_threshold)
  */
  for (size_t coeff_index = 0; coeff_index < reg_count; coeff_index++) {
    for (int e = 0; e < 4; e++) {
      const float cur_coeff = params.reg_params_separte[e][coeff_index];
      const double diff = cur_coeff - params.last_coeffcients[e];
      double itvNum = fabs(diff) / coefficient_sizes.precision[e] + 1;
      if (itvNum < coefficient_sizes.coeff_intvCapacity_sz) {
        if (diff < 0)
          itvNum = -itvNum;
        params.coeff_type[e][coeff_index] =
          (int)(itvNum / 2) + coefficient_sizes.coeff_intvRadius;
        params.last_coeffcients[e] =
          params.last_coeffcients[e] + 2 *
                                         (params.coeff_type[e][coeff_index] -
                                          coefficient_sizes.coeff_intvRadius) *
                                         coefficient_sizes.precision[e];
        // ganrantee compression error against the case of machine-epsilon
        if (fabs(cur_coeff - params.last_coeffcients[e]) >
            coefficient_sizes.precision[e]) {
          params.coeff_type[e][coeff_index] = 0;
          params.last_coeffcients[e] = cur_coeff;
          params.coeff_unpred_data[e][params.coeff_unpredictable_count[e]++] =
            cur_coeff;
        }
      } else {
        params.coeff_type[e][coeff_index] = 0;
        params.last_coeffcients[e] = cur_coeff;
        params.coeff_unpred_data[e][params.coeff_unpredictable_count[e]++] =
          cur_coeff;
      }
      params.reg_params_separte[e][coeff_index] = params.last_coeffcients[e];
    }
  }
  return params;
}

void
decompress_all_blocks(const sz_opencl_sizes &sizes,
                      double realPrecision,
                      float mean,
                      unsigned char use_mean,
                      const unsigned char *indicator,
                      const float *reg_params,
                      int intvRadius,
                      const size_t *unpred_offset,
                      const float *unpred_data,
                      const sz_opencl_decompress_positions &pos,
                      const int *result_type,
                      float *&dec_block_data,
                      size_t unpred_data_size)
{
  dec_block_data =
    (float*)calloc(sizeof(float), pos.dec_block_data_size);

#if SZ_OPENCL_USE_CUDA
  //currently this kernel requires too many resources on the GPU disable it for now
  //decompress_all_blocks_host(&sizes, realPrecision, mean, use_mean, indicator,
  //    reg_params, intvRadius, unpred_offset, unpred_data, &pos, result_type,
  //    dec_block_data, unpred_data_size);
#endif

  //TODO there is a data dependancy on one of the pointers passed
  #pragma omp parallel for collapse(3)
  for (size_t i = pos.start_block1; i < pos.end_block1; i++) {
    for (size_t j = pos.start_block2; j < pos.end_block2; j++) {
      for (size_t k = pos.start_block3; k < pos.end_block3; k++) {
        float* data_pos = dec_block_data + (i - pos.start_block1)*sizes.block_size*pos.dec_block_dim0_offset + (j - pos.start_block2)*sizes.block_size*pos.dec_block_dim1_offset + (k - pos.start_block3)*sizes.block_size; 
        const int* type = result_type +
               (i - pos.start_block1) * sizes.block_size * sizes.block_size *
                 (pos.num_data_blocks2) * sizes.block_size *
                 (pos.num_data_blocks3) +
               (j - pos.start_block2) * sizes.max_num_block_elements *
                 (pos.num_data_blocks3) +
               (k - pos.start_block3) * sizes.max_num_block_elements;
        size_t coeff_index =
          i * sizes.num_y * sizes.num_z + j * sizes.num_z + k;
        const float* block_unpred = unpred_data + unpred_offset[coeff_index];
        if (indicator[coeff_index]) {
          // decompress by SZ
          decompress_location_using_sz(sizes, pos, realPrecision, mean, use_mean,
                                       intvRadius, type, data_pos,
                                       block_unpred);
        } else {
          // decompress by regression
          decompress_location_using_regression(
            sizes, pos, reg_params + 4 * coeff_index, type, block_unpred,
            realPrecision, intvRadius, data_pos);
        }

        // mv data back
        // move_data_block(sizes, pos, data_pos, block_data_pos_x,
        //                 block_data_pos_y, block_data_pos_z, dec_block_data, i,
        //                 j, k);
      }
    }
  }
}

#if !SZ_OPENCL_USE_CUDA
namespace {
  const char*
  getenv_or(const char* env, const char* default_value)
  {
    const char* env_value = getenv(env);
    if(env_value == nullptr) return default_value;
    else return env_value;
  }
}
#endif

void copy_block_data(float **data,
                     const sz_opencl_sizes &sizes,
                     const sz_opencl_decompress_positions &pos,
                     const float *dec_block_data) {// extract data

#if SZ_OPENCL_USE_CUDA
  copy_block_data_host(data, pos, dec_block_data);
#else

  *data = (float*)malloc(sizeof(cl_float) * pos.data_buffer_size);


  //TODO parallelize this loop
#pragma omp parallel for collapse(2)
  for (cl_ulong i = 0; i < pos.data_elms1; i++) {
    for (cl_ulong j = 0; j < pos.data_elms2; j++) {
      const float* block_data_pos = dec_block_data + (i + pos.resi_x) * pos.dec_block_dim0_offset + (j + pos.resi_y) * pos.dec_block_dim1_offset + pos.resi_z;
      float* final_data_pos = *data + i*pos.data_elms2*pos.data_elms3 + j * pos.data_elms3;
      for (cl_ulong k = 0; k < pos.data_elms3; k++) {
        *(final_data_pos++) = *(block_data_pos++);
      }
    }
  }
#endif
}

void prepare_data_buffer(const float *oriData, const sz_opencl_sizes &sizes, cl_float *data_buffer) {
#if SZ_OPENCL_USE_CUDA
  std::vector<float> oriData_cuda(oriData, oriData+sizes.num_elements);
  std::vector<float> data_buffer_cuda(sizes.num_blocks * sizes.max_num_block_elements);

  prepare_data_buffer_host(oriData_cuda.data(), &sizes, data_buffer_cuda.data());
#endif

#pragma omp parallel for collapse(3)
  for (cl_ulong i = 0; i < sizes.num_x; i++) {
      for (cl_ulong j = 0; j < sizes.num_y; j++) {
        for (cl_ulong k = 0; k < sizes.num_z; k++) {
          unsigned int block_id = i * (sizes.num_y * sizes.num_z) + j * sizes.num_z + k;
          cl_float* data_buffer_location = data_buffer + block_id * sizes.max_num_block_elements;
          for(unsigned int ii=0; ii<sizes.block_size; ii++){
            for(unsigned int jj=0; jj<sizes.block_size; jj++){
              for(unsigned int kk=0; kk<sizes.block_size; kk++){
                // index in origin data
                cl_ulong i_ = i * sizes.block_size + ii;
                cl_ulong j_ = j * sizes.block_size + jj;
                cl_ulong k_ = k * sizes.block_size + kk;
                i_ = (i_ < sizes.r1) ? i_ : sizes.r1 - 1;
                j_ = (j_ < sizes.r2) ? j_ : sizes.r2 - 1;
                k_ = (k_ < sizes.r3) ? k_ : sizes.r3 - 1;
                data_buffer_location[ii * sizes.block_size * sizes.block_size + jj * sizes.block_size + kk] = oriData[i_ * sizes.r2 * sizes.r3 + j_ * sizes.r3 + k_];
              }
            }
          }
        }
      }
    }
}

extern "C"
{
  int sz_opencl_init(struct sz_opencl_state** state)
  {
    try {

			*state = new sz_opencl_state;

#if !SZ_OPENCL_USE_CUDA
      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);
      std::string desired_platform(getenv_or("SZ_CL_PLATFORM",""));
      std::string desired_device(getenv_or("SZ_CL_DEVICE",""));
      (*state)->debug_level = atoi(getenv_or("SZ_CL_DEBUG", "0"));

      auto valid_platform =
        std::find_if(std::begin(platforms), std::end(platforms),
                     [state, &desired_platform, &desired_device](cl::Platform const& platform) {
                       try {
                         std::vector<cl::Device> devices;
                         platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
                         auto device_it = std::find_if(std::begin(devices), std::end(devices),
                             [state,&platform,&desired_platform, &desired_device](cl::Device const& device){ 
                             auto platform_name = platform.getInfo<CL_PLATFORM_NAME>();
                             auto device_name = device.getInfo<CL_DEVICE_NAME>();
                             if((*state)->debug_level) {
                              printf("%s %s\n", platform_name.c_str(), device_name.c_str());
                             }
                             return (platform_name.find(desired_platform) != std::string::npos) &&
                                    (device_name.find(desired_device) != std::string::npos);
                         });
                         if(device_it == std::end(devices))
                         {
                            return false;
                         }
                         (*state)->device = *device_it;
                         (*state)->platform = platform;
                         return true;
                       } catch (cl::Error const& error) {
                         if (error.err() != CL_DEVICE_NOT_FOUND)
                           throw;
                       }
                       return false;
                     });
      if (valid_platform == std::end(platforms))
        throw cl::Error(CL_DEVICE_NOT_FOUND, "Failed to find a GPU");

      (*state)->context = cl::Context(std::vector<cl::Device>({(*state)->device }));
      (*state)->queue = cl::CommandQueue((*state)->context, (*state)->device);
      auto sources = get_sz_kernel_sources();
      cl::Program program((*state)->context, sources);
      try {
      program.build({ (*state)->device }, "-I " SZ_OPENCL_KERNEL_INCLUDE_DIR " " SZ_OPENCL_KERNEL_CFLAGS);
      (*state)->calculate_regression_coefficents =
        cl::Kernel(program, "calculate_regression_coefficents");
      } catch (cl::Error const& cl_error) {
        if(cl_error.err() == CL_BUILD_PROGRAM_FAILURE) {
          (*state)->error.code = cl_error.err();
          (*state)->error.str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>((*state)->device);
          return SZ_NSCS;
        } else {
          throw;
        }
      }
#endif

      return SZ_SCES;
#if !SZ_OPENCL_USE_CUDA
    } catch (cl::Error const& cl_error) {
      (*state)->error.code = cl_error.err();
      (*state)->error.str = cl_error.what();
      return SZ_NSCS;
#endif
    } catch (sz_opencl_exception const& sz_error) {
      (*state)->error.code = -1;
      (*state)->error.str = sz_error.what();
      return SZ_NSCS;
    } catch (...) {
      delete *state;
      *state = nullptr;
      return SZ_NSCS;
    }
  }

  int sz_opencl_release(struct sz_opencl_state** state)
  {
    delete *state;

    return SZ_SCES;
  }

  const char* sz_opencl_error_msg(struct sz_opencl_state* state)
  {
    if (state == nullptr) {
      return "sz opencl allocation failed";
    }

    return state->error.str.c_str();
  }

  int sz_opencl_error_code(struct sz_opencl_state* state)
  {
    if (state == nullptr) {
      return -1;
    }

    return state->error.code;
  }

  int sz_opencl_check(struct sz_opencl_state* state)
  {
#if !SZ_OPENCL_USE_CUDA
    try {
      std::string vec_add(
        R"(
				kernel void add(__global float* a, __global float* b, __global float* c)
				{
					int id = get_global_id(0);
					c[id] = a[id] + b[id];
				}
				)");
      cl::Program::Sources sources(
        1, std::make_pair(vec_add.c_str(), vec_add.size() + 1));

      cl::Program program(state->context, sources);
      program.build({ state->device });
      cl::Kernel kernel(program, "add");
      const int size = 1024;
      std::vector<float> h_a(size);
      std::vector<float> h_b(size);
      std::vector<float> h_c(size);
      std::vector<float> verify(size);
      cl::Buffer d_a(state->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     sizeof(cl_float) * size);
      cl::Buffer d_b(state->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     sizeof(cl_float) * size);
      cl::Buffer d_c(state->context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     sizeof(cl_float) * size);

      auto random_fill = [](std::vector<float>& vec, int seed) {
        std::seed_seq seed_seq{ seed };
        std::mt19937 gen(seed_seq);
        std::uniform_real_distribution<float> dist;

        std::generate(std::begin(vec), std::end(vec),
                      [&dist, &gen]() { return dist(gen); });
      };
      random_fill(h_a, 0);
      random_fill(h_b, 1);
      random_fill(h_c, 2);
      std::transform(std::begin(h_a), std::end(h_a), std::begin(h_b),
                     std::begin(verify),
                     [](float a, float b) { return a + b; });

      kernel.setArg(0, d_a);
      kernel.setArg(1, d_b);
      kernel.setArg(2, d_c);

      state->queue.enqueueWriteBuffer(d_a, CL_BLOCKING, /*offset*/ 0,
                                      /*size*/ sizeof(cl_float) * size,
                                      h_a.data());
      state->queue.enqueueWriteBuffer(d_b, CL_BLOCKING, /*offset*/ 0,
                                      /*size*/ sizeof(cl_float) * size,
                                      h_b.data());
      state->queue.enqueueNDRangeKernel(kernel, /*offset*/ cl::NullRange,
                                        /*global*/ cl::NDRange(size),
                                        cl::NullRange);
      state->queue.finish();
      state->queue.enqueueReadBuffer(d_c, CL_BLOCKING, /*offset*/ 0,
                                     /*size*/ sizeof(cl_float) * size,
                                     h_c.data());

      if (std::equal(std::begin(h_c), std::end(h_c), std::begin(verify))) {
        return SZ_SCES;
      } else {
        return SZ_NSCS;
      }
    } catch (cl::Error const& error) {
      state->error.code = error.err();
      state->error.str = error.what();
      return SZ_NSCS;
    }
#else 
		return SZ_SCES;
#endif
  }

  unsigned char* sz_compress_float3d_opencl(struct sz_opencl_state* state,
                                            float* oriData, size_t r1,
                                            size_t r2, size_t r3,
                                            cl_double realPrecision,
                                            size_t* comp_size)
  {
    unsigned int quantization_intervals;
    bool use_mean = false;

    // calculate block dims
    const sz_opencl_sizes sizes = make_sz_opencl_sizes(/*block_size*/ 6, r1, r2, r3);

    int* result_type = (int*)malloc(sizes.data_buffer_size * sizeof(int));
    cl_float* result_unpredictable_data = (cl_float*)malloc(
      sizes.unpred_data_max_size * sizeof(cl_float) * sizes.num_blocks);
    cl_float* reg_params =
      (cl_float*)malloc(sizeof(cl_float) * sizes.reg_params_buffer_size);
    cl_float* data_buffer =
      (cl_float*)malloc(sizeof(cl_float) * sizes.data_buffer_size);

    prepare_data_buffer(oriData, sizes, data_buffer);

    unsigned char* indicator =
      (unsigned char*)calloc(sizes.num_blocks, sizeof(unsigned char));
    int* blockwise_unpred_count = (int*)malloc(sizes.num_blocks * sizeof(int));

    int* type = result_type;
    int* blockwise_unpred_count_pos = blockwise_unpred_count;

    calculate_regression_coefficents(state, oriData, &sizes, reg_params, data_buffer);

    float mean = 0.0f;
    if (exe_params->optQuantMode == 1) {
      float mean_flush_freq, dense_pos, sz_sample_correct_freq = -1; // 0.5;
                                                                     // //-1
      quantization_intervals =
        optimize_intervals_float_3D_with_freq_and_dense_pos(
          oriData, r1, r2, r3, realPrecision, &dense_pos,
          &sz_sample_correct_freq, &mean_flush_freq);
      if (mean_flush_freq > 0.5 || mean_flush_freq > sz_sample_correct_freq) {
        use_mean = 1;
        mean =
          compute_mean(oriData, realPrecision, dense_pos, sizes.num_elements);
      }
      updateQuantizationInfo(quantization_intervals);
    } else {
      quantization_intervals = exe_params->intvCapacity;
    }

    // use two prediction buffers for higher performance
    float* unpredictable_data = result_unpredictable_data;
    unsigned char* indicator_pos = indicator;

    float noise = realPrecision * 1.22;
    float* reg_params_pos = reg_params;

    // select
    opencl_sample(&sizes, mean, noise, use_mean, data_buffer,
                  reg_params_pos, indicator_pos);

    size_t reg_count = 0;
    for (size_t i = 0; i < sizes.num_blocks; i++) {
      if (!(indicator[i])) {
        reg_params[reg_count] = reg_params[i];
        reg_params[reg_count + sizes.params_offset_b] =
          reg_params[i + sizes.params_offset_b];
        reg_params[reg_count + sizes.params_offset_c] =
          reg_params[i + sizes.params_offset_c];
        reg_params[reg_count + sizes.params_offset_d] =
          reg_params[i + sizes.params_offset_d];
        reg_count++;
      }
    }

    // Compress coefficient arrays
    const sz_opencl_coefficient_sizes coefficient_sizes(realPrecision,
                                                        sizes.block_size);
    sz_opencl_coefficient_params params(
      reg_count, sizes.num_blocks, reg_params,
      /*coeff_result_type=*/(cl_int*)malloc(reg_count * 4 * sizeof(cl_int)),
      /*coeff_unpredictable_data=*/
      (cl_float*)malloc(reg_count * 4 * sizeof(cl_float)));

    params = compress_coefficent_arrays(reg_count, coefficient_sizes, params);

    // pred & quantization
    //reg_params_pos = reg_params;
    indicator_pos = indicator;

    size_t total_unpred = save_unpredictable_data(
        &sizes, realPrecision, mean, use_mean,
        data_buffer, unpredictable_data, result_type, reg_params,
        indicator_pos, blockwise_unpred_count_pos);

    free(data_buffer);
    int stateNum = 2 * quantization_intervals;
    HuffmanTree* huffmanTree = createHuffmanTree(stateNum);

    size_t nodeCount = 0;
    init(huffmanTree, result_type,
         sizes.num_blocks * sizes.max_num_block_elements);
    size_t i = 0;
    for (i = 0; i < huffmanTree->stateNum; i++)
      if (huffmanTree->code[i])
        nodeCount++;
    nodeCount = nodeCount * 2 - 1;

    unsigned char* treeBytes;
    unsigned int treeByteSize =
      convert_HuffTree_to_bytes_anyStates(huffmanTree, nodeCount, &treeBytes);

    const unsigned int meta_data_offset = 3 + 1 + MetaDataByteLength;
    // total size 										metadata
    // # elements real precision intervals nodeCount		huffman block
    // index unpredicatable count mean unpred size elements
    unsigned char* result = (unsigned char*)calloc(
      meta_data_offset + exe_params->SZ_SIZE_TYPE + sizeof(double) +
        sizeof(int) + sizeof(int) + treeByteSize +
        sizes.num_blocks * sizeof(unsigned short) +
        sizes.num_blocks * sizeof(unsigned short) +
        sizes.num_blocks * sizeof(float) + total_unpred * sizeof(float) +
        sizes.num_elements * sizeof(int),
      1);
    unsigned char* result_pos = result;
    initRandomAccessBytes(result_pos);

    result_pos += meta_data_offset;

    sizeToBytes(result_pos, sizes.num_elements); // SZ_SIZE_TYPE: 4 or 8
    result_pos += exe_params->SZ_SIZE_TYPE;

    intToBytes_bigEndian(result_pos, sizes.block_size);
    result_pos += sizeof(int);
    doubleToBytes(result_pos, realPrecision);
    result_pos += sizeof(double);
    intToBytes_bigEndian(result_pos, quantization_intervals);
    result_pos += sizeof(int);
    intToBytes_bigEndian(result_pos, treeByteSize);
    result_pos += sizeof(int);
    intToBytes_bigEndian(result_pos, nodeCount);
    result_pos += sizeof(int);
    memcpy(result_pos, treeBytes, treeByteSize);
    result_pos += treeByteSize;
    free(treeBytes);

    memcpy(result_pos, &use_mean, sizeof(unsigned char));
    result_pos += sizeof(unsigned char);
    memcpy(result_pos, &mean, sizeof(float));
    result_pos += sizeof(float);
    size_t indicator_size = convertIntArray2ByteArray_fast_1b_to_result(
      indicator, sizes.num_blocks, result_pos);
    result_pos += indicator_size;

    // convert the lead/mid/resi to byte stream
    if (reg_count > 0) {
      for (int e = 0; e < 4; e++) {
        int stateNum = 2 * coefficient_sizes.coeff_intvCapacity_sz;
        HuffmanTree* huffmanTree = createHuffmanTree(stateNum);
        size_t nodeCount = 0;
        init(huffmanTree, params.coeff_type[e], reg_count);
        size_t i = 0;
        for (i = 0; i < huffmanTree->stateNum; i++)
          if (huffmanTree->code[i])
            nodeCount++;
        nodeCount = nodeCount * 2 - 1;
        unsigned char* treeBytes;
        unsigned int treeByteSize = convert_HuffTree_to_bytes_anyStates(
          huffmanTree, nodeCount, &treeBytes);
        doubleToBytes(result_pos, coefficient_sizes.precision[e]);
        result_pos += sizeof(double);
        intToBytes_bigEndian(result_pos, coefficient_sizes.coeff_intvRadius);
        result_pos += sizeof(int);
        intToBytes_bigEndian(result_pos, treeByteSize);
        result_pos += sizeof(int);
        intToBytes_bigEndian(result_pos, nodeCount);
        result_pos += sizeof(int);
        memcpy(result_pos, treeBytes, treeByteSize);
        result_pos += treeByteSize;
        free(treeBytes);
        size_t typeArray_size = 0;
        encode(huffmanTree, params.coeff_type[e], reg_count,
               result_pos + sizeof(size_t), &typeArray_size);
        sizeToBytes(result_pos, typeArray_size);
        result_pos += sizeof(size_t) + typeArray_size;
        intToBytes_bigEndian(result_pos, params.coeff_unpredictable_count[e]);
        result_pos += sizeof(int);
        memcpy(result_pos, params.coeff_unpred_data[e],
               params.coeff_unpredictable_count[e] * sizeof(float));
        result_pos += params.coeff_unpredictable_count[e] * sizeof(float);
        SZ_ReleaseHuffman(huffmanTree);
      }
    }
    free(params.coeff_result_type);
    free(params.coeff_unpredicatable_data);

    // record the number of unpredictable data and also store them
    memcpy(result_pos, &total_unpred, sizeof(size_t));
    result_pos += sizeof(size_t);
    // record blockwise unpred data
    size_t compressed_blockwise_unpred_count_size;
    unsigned char* compressed_bw_unpred_count = SZ_compress_args(
      SZ_INT32, blockwise_unpred_count, &compressed_blockwise_unpred_count_size,
      ABS, 0.5, 0, 0, 0, 0, 0, 0, sizes.num_blocks);
    memcpy(result_pos, &compressed_blockwise_unpred_count_size, sizeof(size_t));
    result_pos += sizeof(size_t);
    memcpy(result_pos, compressed_bw_unpred_count,
           compressed_blockwise_unpred_count_size);
    result_pos += compressed_blockwise_unpred_count_size;
    free(blockwise_unpred_count);
    free(compressed_bw_unpred_count);
    memcpy(result_pos, result_unpredictable_data, total_unpred * sizeof(float));
    result_pos += total_unpred * sizeof(float);

    free(reg_params);
    free(indicator);
    free(result_unpredictable_data);
    // encode type array by block

    result_pos =
      encode_all_blocks(&sizes, result_type, type, huffmanTree, result_pos);

    size_t totalEncodeSize = result_pos - result;

    free(result_type);
    SZ_ReleaseHuffman(huffmanTree);
    *comp_size = totalEncodeSize;
    return result;
  }

  void sz_decompress_float_opencl_impl(float** data, size_t r1, size_t r2,
                                       size_t r3, size_t s1, size_t s2,
                                       size_t s3, size_t e1, size_t e2,
                                       size_t e3, unsigned char* comp_data)
  {

    // size_t dim0_offset = r2 * r3;
    // size_t dim1_offset = r3;

    unsigned char* comp_data_pos = comp_data;

    const sz_opencl_sizes sizes = make_sz_opencl_sizes(bytesToInt_bigEndian(comp_data_pos), r1, r2, r3);
    comp_data_pos += sizeof(int);
    // calculate block dims

    double realPrecision = bytesToDouble(comp_data_pos);
    comp_data_pos += sizeof(double);
    unsigned int intervals = bytesToInt_bigEndian(comp_data_pos);
    comp_data_pos += sizeof(int);

    updateQuantizationInfo(intervals);

    unsigned int tree_size = bytesToInt_bigEndian(comp_data_pos);
    comp_data_pos += sizeof(int);

    int stateNum = 2 * intervals;
    HuffmanTree* huffmanTree = createHuffmanTree(stateNum);

    int nodeCount = bytesToInt_bigEndian(comp_data_pos);
    node root = reconstruct_HuffTree_from_bytes_anyStates(
      huffmanTree, comp_data_pos + sizeof(int), nodeCount);
    comp_data_pos += sizeof(int) + tree_size;

    float mean;
    unsigned char use_mean;
    memcpy(&use_mean, comp_data_pos, sizeof(unsigned char));
    comp_data_pos += sizeof(unsigned char);
    memcpy(&mean, comp_data_pos, sizeof(float));
    comp_data_pos += sizeof(float);
    size_t reg_count = 0;

    unsigned char* indicator;
    size_t indicator_bitlength = (sizes.num_blocks - 1) / 8 + 1;
    convertByteArray2IntArray_fast_1b(sizes.num_blocks, comp_data_pos,
                                      indicator_bitlength, &indicator);
    comp_data_pos += indicator_bitlength;
    for (size_t i = 0; i < sizes.num_blocks; i++) {
      if (!indicator[i])
        reg_count++;
    }

    int coeff_intvRadius[4];
    int* coeff_result_type = (int*)malloc(sizes.num_blocks * 4 * sizeof(int));
    int* coeff_type[4];
    double precision[4];
    float* coeff_unpred_data[4];
    if (reg_count > 0) {
      for (int i = 0; i < 4; i++) {
        precision[i] = bytesToDouble(comp_data_pos);
        comp_data_pos += sizeof(double);
        coeff_intvRadius[i] = bytesToInt_bigEndian(comp_data_pos);
        comp_data_pos += sizeof(int);
        unsigned int tree_size = bytesToInt_bigEndian(comp_data_pos);
        comp_data_pos += sizeof(int);
        int stateNum = 2 * coeff_intvRadius[i] * 2;
        HuffmanTree* huffmanTree = createHuffmanTree(stateNum);
        int nodeCount = bytesToInt_bigEndian(comp_data_pos);
        node root = reconstruct_HuffTree_from_bytes_anyStates(
          huffmanTree, comp_data_pos + sizeof(int), nodeCount);
        comp_data_pos += sizeof(int) + tree_size;

        coeff_type[i] = coeff_result_type + i * sizes.num_blocks;
        size_t typeArray_size = bytesToSize(comp_data_pos);
        decode(comp_data_pos + sizeof(size_t), reg_count, root, coeff_type[i]);
        comp_data_pos += sizeof(size_t) + typeArray_size;
        int coeff_unpred_count = bytesToInt_bigEndian(comp_data_pos);
        comp_data_pos += sizeof(int);
        coeff_unpred_data[i] = (float*)comp_data_pos;
        comp_data_pos += coeff_unpred_count * sizeof(float);
        SZ_ReleaseHuffman(huffmanTree);
      }
    }

    float last_coefficients[4] = { 0.0 };
    int coeff_unpred_data_count[4] = { 0 };

    float* reg_params = (float*)calloc(sizeof(float), 4 * sizes.num_blocks);
    decompress_coefficents(sizes, indicator, coeff_intvRadius, coeff_type,
                           precision, coeff_unpred_data, last_coefficients,
                           coeff_unpred_data_count, reg_params);

    updateQuantizationInfo(intervals);
    int intvRadius = exe_params->intvRadius;

    size_t total_unpred;
    memcpy(&total_unpred, comp_data_pos, sizeof(size_t));
    comp_data_pos += sizeof(size_t);
    size_t compressed_blockwise_unpred_count_size;
    memcpy(&compressed_blockwise_unpred_count_size, comp_data_pos,
           sizeof(size_t));
    comp_data_pos += sizeof(size_t);
    int* blockwise_unpred_count = NULL;
    SZ_decompress_args_int32(&blockwise_unpred_count, 0, 0, 0, 0,
                             sizes.num_blocks, comp_data_pos,
                             compressed_blockwise_unpred_count_size);
    comp_data_pos += compressed_blockwise_unpred_count_size;
    size_t* unpred_offset = (size_t*)malloc(sizes.num_blocks * sizeof(size_t));
    size_t cur_offset = 0;
    for (size_t i = 0; i < sizes.num_blocks; i++) {
      unpred_offset[i] = cur_offset;
      cur_offset += blockwise_unpred_count[i];
    }

    float* unpred_data = (float*)comp_data_pos;
    size_t unpred_data_size = sizeof(float) * total_unpred;
    comp_data_pos += unpred_data_size;

    size_t compressed_type_array_block_size;
    memcpy(&compressed_type_array_block_size, comp_data_pos, sizeof(size_t));
    comp_data_pos += sizeof(size_t);
    unsigned short* type_array_block_size = NULL;
    SZ_decompress_args_uint16(&type_array_block_size, 0, 0, 0, 0,
                              sizes.num_blocks, comp_data_pos,
                              compressed_type_array_block_size);

    comp_data_pos += compressed_type_array_block_size;

    // compute given area
    sz_opencl_decompress_positions pos(sizes.block_size, s1, s2, s3, e1, e2,
                                       e3);

    unsigned short* type_array_block_size_pos = type_array_block_size;
    size_t* type_array_offset =
      (size_t*)malloc(sizes.num_blocks * sizeof(size_t));
    size_t* type_array_offset_pos = type_array_offset;
    size_t cur_type_array_offset = 0;
    //TODO parallelize this loop
    //#pragma omp parallel for collapse(3)
    //(Prefix_Sum: To compute the sum of the first value through current index)
    //We believe Robert can do it better than us. - Xin and Sheng
    for (size_t i = 0; i < sizes.num_x; i++) {
      for (size_t j = 0; j < sizes.num_y; j++) {
        for (size_t k = 0; k < sizes.num_z; k++) {
          *(type_array_offset_pos++) = cur_type_array_offset;
          cur_type_array_offset += *(type_array_block_size_pos++);
        }
      }
    }
    free(type_array_block_size);
    int* result_type = (int*)malloc(pos.dec_block_data_size* sizeof(int));
    decode_all_blocks(comp_data_pos, sizes, root, pos, type_array_offset,
                      result_type);
    SZ_ReleaseHuffman(huffmanTree);
    free(type_array_offset);

    float* dec_block_data;
    decompress_all_blocks(sizes,
                          realPrecision,
                          mean,
                          use_mean,
                          indicator,
                          reg_params,
                          intvRadius,
                          unpred_offset,
                          unpred_data,
                          pos,
                          result_type,
                          dec_block_data,
                          unpred_data_size);

    free(unpred_offset);
    free(reg_params);
    free(blockwise_unpred_count);
    free(coeff_result_type);

    free(indicator);
    free(result_type);

    copy_block_data(data, sizes, pos, dec_block_data);

    free(dec_block_data);
  }

  int sz_decompress_float_opencl(struct sz_opencl_state *state,
                                   float **newData,
                                   size_t r5,
                                   size_t r4,
                                   size_t r3,
                                   size_t r2,
                                   size_t r1,
                                   size_t s5,
                                   size_t s4,
                                   size_t s3,
                                   size_t s2,
                                   size_t s1,
                                   size_t e5,
                                   size_t e4,
                                   size_t e3,
                                   size_t e2,
                                   size_t e1,
                                   unsigned char *cmpBytes,
                                   size_t cmpSize)
  {
    if (confparams_dec == NULL)
      confparams_dec = (sz_params*)malloc(sizeof(sz_params));
    memset(confparams_dec, 0, sizeof(sz_params));
    if (exe_params == NULL)
      exe_params = (sz_exedata*)malloc(sizeof(sz_exedata));
    memset(exe_params, 0, sizeof(sz_exedata));

    int x = 1;
    char* y = (char*)&x;
    if (*y == 1)
      sysEndianType = LITTLE_ENDIAN_SYSTEM;
    else //=0
      sysEndianType = BIG_ENDIAN_SYSTEM;

    confparams_dec->randomAccess = 1;

    int status = SZ_SCES;
    size_t dataLength = computeDataLength(r5, r4, r3, r2, r1);

    // unsigned char* tmpBytes;
    size_t targetUncompressSize = dataLength << 2; // i.e., *4
    // tmpSize must be "much" smaller than dataLength
    size_t i, tmpSize = 8 + MetaDataByteLength + exe_params->SZ_SIZE_TYPE;
    unsigned char* szTmpBytes;

    if (cmpSize != 8 + 4 + MetaDataByteLength &&
        cmpSize !=
          8 + 8 +
            MetaDataByteLength) // 4,8 means two posibilities of SZ_SIZE_TYPE
    {
      confparams_dec->losslessCompressor =
        is_lossless_compressed_data(cmpBytes, cmpSize);
      if (confparams_dec->szMode != SZ_TEMPORAL_COMPRESSION) {
        if (confparams_dec->losslessCompressor != -1)
          confparams_dec->szMode = SZ_BEST_COMPRESSION;
        else
          confparams_dec->szMode = SZ_BEST_SPEED;
      }

      if (confparams_dec->szMode == SZ_BEST_SPEED) {
        tmpSize = cmpSize;
        szTmpBytes = cmpBytes;
      } else if (confparams_dec->szMode == SZ_BEST_COMPRESSION ||
                 confparams_dec->szMode == SZ_DEFAULT_COMPRESSION ||
                 confparams_dec->szMode == SZ_TEMPORAL_COMPRESSION) {
        if (targetUncompressSize <
            MIN_ZLIB_DEC_ALLOMEM_BYTES) // Considering the minimum size
          targetUncompressSize = MIN_ZLIB_DEC_ALLOMEM_BYTES;
        tmpSize = sz_lossless_decompress(
          confparams_dec->losslessCompressor, cmpBytes, (unsigned long)cmpSize,
          &szTmpBytes,
          (unsigned long)targetUncompressSize + 4 + MetaDataByteLength +
            exe_params
              ->SZ_SIZE_TYPE); //		(unsigned
                               // long)targetUncompressSize+8: consider the
                               // total length under lossless compression mode
                               // is actually 3+4+1+targetUncompressSize
      } else {
        printf("Wrong value of confparams_dec->szMode in the double compressed "
               "bytes.\n");
        status = SZ_MERR;
        return status;
      }
    } else
      szTmpBytes = cmpBytes;

    TightDataPointStorageF* tdps;
    new_TightDataPointStorageF_fromFlatBytes(&tdps, szTmpBytes, tmpSize);

    int dim = computeDimension(r5, r4, r3, r2, r1);
    int floatSize = sizeof(float);
    if (tdps->isLossless) {
      *newData = (float*)malloc(floatSize * dataLength);
      if (sysEndianType == BIG_ENDIAN_SYSTEM) {
        memcpy(*newData,
               szTmpBytes + 4 + MetaDataByteLength + exe_params->SZ_SIZE_TYPE,
               dataLength * floatSize);
      } else {
        unsigned char* p =
          szTmpBytes + 4 + MetaDataByteLength + exe_params->SZ_SIZE_TYPE;
        for (i = 0; i < dataLength; i++, p += floatSize)
          (*newData)[i] = bytesToFloat(p);
      }
    } else {
      if (confparams_dec->randomAccess == 0 &&
          (s1 + s2 + s3 + s4 + s5 > 0 ||
           (r5 - e5 + r4 - e4 + r3 - e3 + r2 - e2 + r1 - e1 > 0))) {
        printf(
          "Error: you specified the random access mode for decompression, but "
          "the compressed data were generate in the non-random-access way.!\n");
        status = SZ_DERR;
      } else if (dim == 1) {
        printf(
          "Error: random access mode doesn't support 1D yet, but only 3D.\n");
        status = SZ_DERR;
      } else if (dim == 2) {
        printf(
          "Error: random access mode doesn't support 2D yet, but only 3D.\n");
        status = SZ_DERR;
      } else if (dim == 3) {
        sz_decompress_float_opencl_impl(newData, r3, r2, r1, s3, s2, s1, e3, e2,
                                        e1, tdps->raBytes);
        status = SZ_SCES;
      } else if (dim == 4) {
        printf(
          "Error: random access mode doesn't support 4D yet, but only 3D.\n");
        status = SZ_DERR;
      } else {
        printf("Error: currently support only at most 4 dimensions!\n");
        status = SZ_DERR;
      }
    }

    free_TightDataPointStorageF2(tdps);
    if (confparams_dec->szMode != SZ_BEST_SPEED &&
        cmpSize != 8 + MetaDataByteLength + exe_params->SZ_SIZE_TYPE)
      free(szTmpBytes);
    return status;
  }
}



struct sz_opencl_sizes
make_sz_opencl_sizes(cl_ulong block_size, cl_ulong r1, cl_ulong r2, cl_ulong r3)
{
  struct sz_opencl_sizes sizes;
  sizes.r1=r1;
  sizes.r2=r2;
  sizes.r3=r3;
  sizes.block_size=block_size;
  sizes.num_x=(sizes.r1 - 1) / block_size + 1;
  sizes.num_y=(sizes.r2 - 1) / block_size + 1;
  sizes.num_z=(sizes.r3 - 1) / block_size + 1;
  sizes.max_num_block_elements=block_size * block_size * block_size;
  sizes.num_blocks=sizes.num_x * sizes.num_y * sizes.num_z;
  sizes.num_elements=r1 * r2 * r3;
  sizes.dim0_offset=r2 * r3;
  sizes.dim1_offset=r3;
  sizes.params_offset_b=sizes.num_blocks;
  sizes.params_offset_c=2 * sizes.num_blocks;
  sizes.params_offset_d=3 * sizes.num_blocks;
  sizes.pred_buffer_block_size=sizes.block_size;
  sizes.strip_dim0_offset=sizes.pred_buffer_block_size * sizes.pred_buffer_block_size;
  sizes.strip_dim1_offset=sizes.pred_buffer_block_size;
  sizes.unpred_data_max_size=sizes.max_num_block_elements;
  sizes.reg_params_buffer_size=sizes.num_blocks * 4;
  sizes.pred_buffer_size=sizes.num_blocks * sizes.num_blocks;
  sizes.block_dim0_offset=sizes.pred_buffer_block_size*sizes.pred_buffer_block_size;
  sizes.block_dim1_offset=sizes.pred_buffer_block_size;
  sizes.data_buffer_size=sizes.num_blocks * sizes.max_num_block_elements;
  return sizes;
}
