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
#include "sz_opencl_kernels.h"

#define SOLO(actions)                                                          \
  do {                                                                         \
    cl_ulong __dim = get_work_dim();                                           \
    if (((__dim >= 1) ? get_global_id(0) == 0 : 1) &&                          \
        ((__dim >= 2) ? get_global_id(1) == 0 : 1) &&                          \
        ((__dim >= 3) ? get_global_id(2) == 0 : 1)) {                          \
      actions                                                                  \
    }                                                                          \
    barrier(CLK_GLOBAL_MEM_FENCE);                                             \
  } while (0);

void print_opencl_sizes_debug(struct sz_opencl_sizes const*const  sizes)  {
      printf("little endian %o\n", __ENDIAN_LITTLE__);                       
      printf("r1 %d\n", (sizes)->r1); printf("r2 %d\n", (sizes)->r2);          
      printf("r3 %d\n", (sizes)->r3);                                         
      printf("block_size %d\n", (sizes)->block_size);                         
      printf("num_x %d\n", (sizes)->num_x);                                   
      printf("num_y %d\n", (sizes)->num_y);                                   
      printf("num_z %d\n", (sizes)->num_z);                                   
      printf("max_num_block_elements %d\n", (sizes)->max_num_block_elements); 
      printf("num_blocks %d\n", (sizes)->num_blocks);                         
      printf("num_elements %d\n", (sizes)->num_elements);                     
      printf("dim0_offset %d\n", (sizes)->dim0_offset);                       
      printf("dim1_offset %d\n", (sizes)->dim1_offset);                       
      printf("params_offset_b %d\n", (sizes)->params_offset_b);               
      printf("params_offset_c %d\n", (sizes)->params_offset_c);               
      printf("params_offset_d %d\n", (sizes)->params_offset_d);               
      printf("pred_buffer_block_size %d\n", (sizes)->pred_buffer_block_size); 
      printf("strip_dim0_offset %d\n", (sizes)->strip_dim0_offset);           
      printf("strip_dim1_offset %d\n", (sizes)->strip_dim1_offset);           
      printf("unpred_data_max_size %d\n", (sizes)->unpred_data_max_size);     
      printf("reg_params_buffer_size %d\n", (sizes)->reg_params_buffer_size); 
      printf("pred_buffer_size %d\n", (sizes)->pred_buffer_size);             
      printf("block_dim0_offset %d\n", (sizes)->block_dim0_offset);           
      printf("block_dim1_offset %d\n", (sizes)->block_dim1_offset);           
      printf("sizeof(sz_opencl_sizes) %d\n", sizeof(struct sz_opencl_sizes));
}

kernel void
print_opencl_data_debug(__global const cl_float* data, __global struct sz_opencl_sizes const* sizes)
{
    print_first_last_3(data, sizes->num_elements);
}

kernel void
calculate_regression_coefficents(
        __global const cl_float* oriData,
        __global struct sz_opencl_sizes const* sizes,
        __global cl_float* reg_params,
        __global cl_float* const pred_buffer)
{
    struct sz_opencl_sizes local_sizes = *sizes;
    cl_ulong i = get_global_id(0);
    cl_ulong j = get_global_id(1);
    cl_ulong k = get_global_id(2);

#ifdef DEBUG
    SOLO(print_opencl_sizes_debug(&local_sizes);
        print_opencl_data_debug(oriData, sizes);
    )

#endif
    const unsigned int block_id =
        i * (sizes->num_y * sizes->num_z) + j * sizes->num_z + k;
    __global const float* local_data_pos = oriData + i * sizes->block_size * sizes->dim0_offset +
        j * sizes->block_size * sizes->dim1_offset + k * sizes->block_size;
    __global float* const pred_buffer_pos = pred_buffer+(block_id*sizes->num_blocks);
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
    __global const float* cur_data_pos = pred_buffer+(block_id*sizes->num_blocks);
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
    __global float* reg_params_pos = reg_params + block_id;
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
  return sizes;
}
