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
#include "sz_opencl_host_utils.h"
#include <cassert>
#include <fstream>
#include <sstream>

#include "sz_opencl.h"
#include "sz_opencl_config.h"
#include "sz_opencl_private.h"

#if !SZ_OPENCL_USE_CUDA
std::string
get_sz_kernel_sources()
{
  std::ifstream infile{ SZ_OPENCL_KERNEL_SOURCE_FILE };
  if (infile.fail())
    throw sz_opencl_exception(
      "missing kernel source: " SZ_OPENCL_KERNEL_SOURCE_FILE);
  std::ostringstream buffer;
  buffer << infile.rdbuf();
  return buffer.str();
}

std::vector<cl::Event>
run_kernel(cl::Kernel kernel, cl::NDRange const& global,
           const sz_opencl_state* state, const sz_opencl_sizes* sizes,
           const std::vector<buffer_copy_info>& buffer_info)
{
  const unsigned int number_of_kernel_args = buffer_info.size();
  assert(kernel.getInfo<CL_KERNEL_NUM_ARGS>() == number_of_kernel_args);
  cl::Event computed;
  std::vector<cl::Event> input_written(number_of_kernel_args);
  std::vector<cl::Event> outputs_written(number_of_kernel_args);
  std::vector<cl::Buffer> buffers(number_of_kernel_args);

  for (size_t i = 0; i < buffer_info.size(); ++i) {
    buffers[i] =
      cl::Buffer(state->context, buffer_info[i].flags, buffer_info[i].size);
    switch (buffer_info[i].copy) {
      case copy_mode::TO:
      case copy_mode::TO_FROM:
        state->queue.enqueueWriteBuffer(
          buffers[i], CL_NON_BLOCKING,
          /*offset*/ 0, buffer_info[i].size, buffer_info[i].ptr,
          /*input events*/ nullptr, &input_written[i]);
        break;
      case copy_mode::ZERO_FROM:
      case copy_mode::ZERO:
        state->queue.enqueueFillBuffer(buffers[i], /*pattern*/ 0, /*offset*/ 0,
                                       buffer_info[i].size, nullptr,
                                       &input_written[i]);
        break;
      case copy_mode::FROM:
        // noop
        break;
    }
    kernel.setArg(i, buffers[i]);
  }

  state->queue.enqueueNDRangeKernel(kernel,
                                    /*offset*/ cl::NullRange,
                                    /*global*/ global,
                                    /*local*/ cl::NullRange, &input_written,
                                    &computed);

  bool waited = false;
  for (size_t i = 0; i < buffer_info.size(); ++i) {
    switch (buffer_info[i].copy) {
      case copy_mode::TO_FROM:
      case copy_mode::FROM:
      case copy_mode::ZERO_FROM: {
        if (!waited) {
          computed.wait();
          waited = true;
        }
        state->queue.enqueueReadBuffer(buffers[i], CL_NON_BLOCKING, 0,
                                       buffer_info[i].size, buffer_info[i].ptr,
                                       nullptr, &outputs_written[i]);
        break;
      }
      case copy_mode::TO:
      case copy_mode::ZERO:
        // noop
        break;
    }
  }

  if (waited)
    return outputs_written;
  else
    return { computed };
}
#endif
