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
/**
 * Contains methods that load or launch kernels
 */
#include <cstddef>
#include <stdexcept>
#include <string>

// forward declare to avoid dependency on sz_opencl.h or sz_opencl_private.h
struct sz_opencl_state;
struct sz_opencl_sizes;

enum class copy_mode
{
  TO,
  FROM,
  ZERO_FROM,
  ZERO,
  TO_FROM
};


class sz_opencl_exception : public std::runtime_error
{
  using std::runtime_error::runtime_error;
};

#if !SZ_OPENCL_USE_CUDA
#include <CL/cl.hpp>
struct buffer_copy_info
{
  void* ptr;
  std::size_t size;
  cl_mem_flags flags;
  copy_mode copy;
};

std::string get_sz_kernel_sources();

std::vector<cl::Event> run_kernel(
  cl::Kernel kernel, cl::NDRange const& global, const sz_opencl_state* state,
  const sz_opencl_sizes* sizes,
  const std::vector<buffer_copy_info>& buffer_info);
#endif
