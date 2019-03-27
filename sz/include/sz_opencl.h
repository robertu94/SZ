//make header C++/C inter-operable
#ifdef __cplusplus
extern "C" {
#endif

#ifndef SZ_OPENCL_H
#define SZ_OPENCL_H

#include<stddef.h>

	//opaque pointer for opencl state
  struct sz_opencl_state;

  /**
   * creates an opencl state for multiple uses of the compressor or
   * returns an error code.
   *
   * \post if return code is SZ_NCES, the state object may only be passed to
   * sz_opencl_release or sz_opencl_error_* otherwise it may be used in any
   * sz_opencl_* function.
   *
   * \param[out] state the sz opencl state
   * \return SZ_SCES for success or SZ_NCES on error
   */
  int sz_opencl_init(struct sz_opencl_state** state);

	/**
	 * deinitializes an opencl state
	 *
	 * \param[in] state the sz opencl state
	 * \return SZ_SCES
	 */
  int sz_opencl_release(struct sz_opencl_state** state);

	/**
	 * returns a human readable error message for the last error recieved by state
	 *
	 * \param[in] state the sz opencl state
	 * \return a pointer to a string that describes the error
	 */
	const char* sz_opencl_error_msg(struct sz_opencl_state* state);


	/**
	 * returns a numeric code for the last error recieved by state
	 *
	 * \param[in] state the sz opencl state
	 * \return the numeric error code
	 */
  int sz_opencl_error_code(struct sz_opencl_state* state);

	/**
	 * confirms that the sz opencl state is ready to use by performing a vector addition
	 *
	 * \param[in] state the sz opencl state
	 * \return SZ_SCES if the opencl implementation is functioning
	 */
	int sz_opencl_check(struct sz_opencl_state*);

	/**
	 * compresses the data using opencl using some algorithm
	 * @param oriData  the data to be compressed
	 * @param r1  the x dimension of the data
	 * @param r2  the y dimension of the data
	 * @param r3  the z dimension of the data
	 * @param comp_size
	 * @return a pointer to the compressed data
	 */
	unsigned char *
	sz_compress_float3d_opencl(struct sz_opencl_state *state, float *oriData, size_t r1, size_t r2, size_t r3,
								   double realPrecision, size_t *comp_size);

  /**
   * decompression scheme using the opencl implementation of SZ
   *
   * @param state the opencl gpu state
   * @param newData  the data that has been decompresed
   * @param r5 the 5th dimension of the data
   * @param r4 the 4th dimension of the data
   * @param r3 the 3th dimension of the data
   * @param r2 the 2th dimension of the data
   * @param r1 the 1th dimension of the data
   * @param s5 the 5th dimension of the starting point of the compression
   * @param s4 the 4th dimension of the starting point of the compression
   * @param s3 the 3th dimension of the starting point of the compression
   * @param s2 the 2th dimension of the starting point of the compression
   * @param s1 the 1th dimension of the starting point of the compression
   * @param e5 the 5th dimension of the starting point of the compression
   * @param e4 the 4th dimension of the starting point of the compression
   * @param e3 the 3th dimension of the starting point of the compression
   * @param e2 the 2th dimension of the starting point of the compression
   * @param e1 the 1th dimension of the starting point of the compression
   * @param cmpBytes the compresed bytes
   * @param cmpSize  the size of the data that has been compressed
   * @return a status code for the compression
   */
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
size_t cmpSize);


#endif /* SZ_OPENCL_H */

//make header C++/C inter-operable
#ifdef __cplusplus
}
#endif
