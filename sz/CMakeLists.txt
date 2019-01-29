aux_source_directory(src sz_sources)

option(DEBUG_CL_KERNELS "Pass Debugging Flags to OpenCL compilers" OFF)
if((CMAKE_BUILD_TYPE MATCHES Debug) AND (DEBUG_CL_KERNELS))
  set(SZ_OPENCL_KERNEL_CFLAGS "-g")
else()
  set(SZ_OPENCL_KERNEL_CFLAGS "")
endif()

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/include/sz_opencl_config.h.in
	${CMAKE_CURRENT_BINARY_DIR}/include/sz_opencl_config.h @ONLY)

add_library (SZ SHARED ${sz_sources})
target_include_directories(SZ PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)
target_link_libraries (SZ zlib zstd m ${OpenCL_LIBRARIES})
target_compile_options(SZ
	PUBLIC -DHAVE_RANDOMACCESS
	PRIVATE $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
          $<$<CONFIG:Release>:-fno-omit-frame-pointer>
	)

install (TARGETS SZ LIBRARY DESTINATION lib)

