cmake_minimum_required (VERSION 3.10.3)
project (sz LANGUAGES C CXX CUDA)
enable_testing()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_DISABLE_SOURCE_CHANGES ON)
set (CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

find_package (FFTW)
if (FFTW_FOUND)
  include_directories (${FFTW_INCLUDES})
  add_definitions (-DHAVE_FFTW3)
endif ()

find_package (NETCDF)
if (NETCDF_FOUND)
  include_directories (${NETCDF_INCLUDE_DIRS})
endif ()

find_package(OpenCL)
if(OpenCL_FOUND)
	include_directories(${OpenCL_INCLUDE_DIRS})
	add_definitions(-DHAVE_OPENCL)
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_program(TAGS ctags)
if(TAGS)
	add_custom_target(tags ALL
    COMMAND ${TAGS} --exclude=${CMAKE_BINARY_DIR} -f ${CMAKE_BINARY_DIR}/tags --c++-kinds=+p --fields=+iaS --extra=+q -R 
		COMMENT Generating Tag files
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		)
endif()

include_directories (
	${CMAKE_CURRENT_SOURCE_DIR}/sz/include 
	${CMAKE_CURRENT_SOURCE_DIR}/sz/include 
	${CMAKE_CURRENT_SOURCE_DIR}/zlib 
	${CMAKE_CURRENT_SOURCE_DIR}/zstd 
	${CMAKE_CURRENT_SOURCE_DIR}/zstd/common 
	${CMAKE_CURRENT_SOURCE_DIR}/zstd/compress
	${CMAKE_CURRENT_SOURCE_DIR}/zstd/decompress
	${CMAKE_CURRENT_SOURCE_DIR}/zstd/deprecated
	${CMAKE_CURRENT_SOURCE_DIR}/zstd/dictBuilder
	${CMAKE_CURRENT_SOURCE_DIR}/zstd/dll
	${CMAKE_CURRENT_SOURCE_DIR}/zstd/legacy
	)
# include_directories (${ZLIB_INCLUDE_DIRS})

add_subdirectory (zlib)
add_subdirectory (zstd)
add_subdirectory (sz)
add_subdirectory (example)
add_subdirectory (test)

