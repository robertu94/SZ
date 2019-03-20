#include <vector>
#include <random>
#include <algorithm>

#include "CUnit/CUnit.h"
#include "CUnit/Basic.h"
#include "CUnit_Array.h"
#include "RegressionTest.hpp"

#include "sz.h"
#include "sz_opencl_config.h"
#include "zlib.h"

namespace {
	struct sz_opencl_state* state = nullptr;
}

extern "C" {
int init_suite()
{
#if !SZ_OPENCL_USE_CUDA
	int rc = sz_opencl_init(&state);
	rc |= sz_opencl_error_code(state);
	const char* msg = sz_opencl_error_msg(state);
	if(rc) fprintf(stderr,"WARNING - %s\n", msg);
	return rc;
#else
	return 0;
#endif
}

int clean_suite()
{
#if !SZ_OPENCL_USE_CUDA
	int rc = sz_opencl_release(&state);
	return rc;
#else
	return 0;
#endif
}


void test_valid_opencl()
{
#if !SZ_OPENCL_USE_CUDA
	int rc = sz_opencl_check(state);
	CU_ASSERT_EQUAL(rc, 0);
#endif
}

void test_identical_opencl_impl()
{
	auto num_random_test_cases = 4;
	auto opencl_compressor = [](float* data, size_t r1, size_t r2, size_t r3,double prec, size_t* out_size){
		return sz_compress_float3d_opencl(state, data, r1, r2, r3, prec, out_size);
	};
	auto opencl_decompressor = sz_decompress_float_opencl;
	auto existing_compressor = SZ_compress_float_3D_MDQ_decompression_random_access_with_blocked_regression;
	auto existing_decompressor = SZ_decompress_args_randomaccess_float;
	test_identical_output_compression_random(num_random_test_cases, opencl_compressor, existing_compressor, existing_decompressor, opencl_decompressor);
	test_identical_output_compression_deterministic(opencl_compressor, existing_compressor, existing_decompressor, opencl_decompressor);
}


int main(int argc, char *argv[])
{
	unsigned int num_failures = 0;
	CU_ErrorCode error_code = CUE_SUCCESS;

	if (CUE_SUCCESS != CU_initialize_registry())
	{
		return CU_get_error();
	}

	CU_pSuite suite = CU_add_suite("test_opencl_suite", init_suite, clean_suite);
	if(suite == nullptr) {
		goto error;
	}

	if(CU_add_test(suite, "test_valid_opencl", test_valid_opencl) == nullptr ||
			CU_add_test(suite, "test_identical_opencl_impl", test_identical_opencl_impl) == nullptr) {
		goto error;
	}

	CU_basic_set_mode(CU_BRM_VERBOSE);
	CU_basic_run_tests();
	error_code = (CU_ErrorCode)(error_code | CU_get_error());
	CU_basic_show_failures(CU_get_failure_list());
	num_failures = CU_get_number_of_failures();

error:
	CU_cleanup_registry();
	error_code = (CU_ErrorCode)(error_code | CU_get_error());
	return  num_failures ||  error_code;
}

}
