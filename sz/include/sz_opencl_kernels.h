#ifndef SZ_OPENCL_KERNELS
#define SZ_OPENCL_KERNELS
#ifdef __cplusplus
#include <CL/cl_platform.h>
#define CL_GLOBAL_DECL
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define cl_ulong ulong
#define cl_float float
#define cl_int int
#define cl_uint uint
#define cl_double double
#define cl_uchar uchar
#define CL_GLOBAL_DECL __global
#endif

struct sz_opencl_sizes
{
  cl_ulong r1, r2, r3;
  cl_ulong block_size;
  cl_uint num_x;
  cl_uint num_y;
  cl_uint num_z;
  cl_ulong max_num_block_elements;
  cl_ulong num_blocks;
  cl_ulong num_elements;
  cl_ulong dim0_offset;
  cl_ulong dim1_offset;
  cl_ulong params_offset_b;
  cl_ulong params_offset_c;
  cl_ulong params_offset_d;
  cl_ulong pred_buffer_block_size;
  cl_ulong strip_dim0_offset;
  cl_ulong strip_dim1_offset;
  cl_ulong unpred_data_max_size;
  cl_ulong reg_params_buffer_size;
  cl_ulong data_buffer_size;
  cl_ulong data_buffer_dim0_offset;
  cl_ulong data_buffer_dim1_offset;

  cl_ulong pred_buffer_size;
  cl_ulong block_dim0_offset;
  cl_ulong block_dim1_offset;
} ;


struct sz_opencl_sizes
make_sz_opencl_sizes(cl_ulong block_size, cl_ulong r1, cl_ulong r2, cl_ulong r3);

struct sz_opencl_decompress_positions {
#ifdef __cplusplus
  sz_opencl_decompress_positions(cl_int block_size, cl_int s1, cl_int s2, cl_int s3, cl_int e1, cl_int e2, cl_int e3)
      : start_elm1(s1),
  start_elm2(s2),
  start_elm3(s3),
  end_elm1(e1),
  end_elm2(e2),
  end_elm3(e3),
  start_block1(s1 / block_size),
  start_block2(s2 / block_size),
  start_block3(s3 / block_size),
  end_block1((e1 - 1) / block_size + 1),
  end_block2((e2 - 1) / block_size + 1),
  end_block3((e3 - 1) / block_size + 1),
  num_data_blocks1(end_block1-start_block1),
  num_data_blocks2(end_block2-start_block2),
  num_data_blocks3(end_block3-start_block3),
  data_elms1(e1-s1),
  data_elms2(e2-s2),
  data_elms3(e3-s3),
  data_buffer_size(data_elms1 * data_elms2 * data_elms3),
  dec_block_dim1_offset(num_data_blocks3 * block_size),
  dec_block_dim0_offset(dec_block_dim1_offset * num_data_blocks2 * block_size)
  {}

#endif
  const cl_ulong start_elm1, start_elm2, start_elm3;
  const cl_ulong end_elm1, end_elm2, end_elm3;
  const cl_ulong start_block1, start_block2, start_block3;
  const cl_ulong end_block1, end_block2, end_block3;
  const cl_ulong num_data_blocks1, num_data_blocks2, num_data_blocks3;
  const cl_ulong data_elms1, data_elms2, data_elms3;
  const cl_ulong data_buffer_size;
  const cl_ulong dec_block_dim1_offset;
  const cl_ulong dec_block_dim0_offset;
};

struct sz_opencl_coefficient_sizes {
#ifdef __cplusplus
  sz_opencl_coefficient_sizes(cl_double realPrecision, cl_ulong block_size):
    rel_param_err(0.025),
    coeff_intvCapacity_sz(65536),
    coeff_intvRadius(coeff_intvCapacity_sz / 2),
    precision{(rel_param_err * realPrecision) / block_size,
              (rel_param_err * realPrecision) / block_size,
              (rel_param_err * realPrecision) / block_size,
              (rel_param_err * realPrecision)}
  {}
#endif
  const cl_float rel_param_err;
  const cl_int coeff_intvCapacity_sz;
  const cl_int coeff_intvRadius;
  const cl_double precision [4];
};


#ifdef __OPENCL_VERSION__
void print_first_last_3(CL_GLOBAL_DECL const cl_float* data, uint size)
{
    printf("%3.2f %3.2f %3.2f %3.2f %3.2f %3.2f\n", 
            data[0],
            data[1],
            data[2],
            data[size-3],
            data[size-2],
            data[size-1]
          );
}
#endif
#endif
