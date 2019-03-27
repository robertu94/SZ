#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>
#include <unistd.h>

namespace chrono = std::chrono;

#include <sz.h>

namespace cmdline {
  struct options {
    size_t r3=0, r2=0, r1=0;
    char const * input_file = nullptr;
    char const * config_file = nullptr;
  };

  const char* usage() {
    return R"("
    ./sz_gpu -c <sz.config> -f <input.f32> [-m [compression][decompress]] r1 r2 r3
    -c config file
    -f input file
    r1,r2,r3 dimentions of file to compression/decompres
  ")"; }

  options parse_options(int argc, char* argv[])
  {
    options options;
    int opt;
    while((opt = getopt(argc, argv, "c:f:") == -1))
    {
      switch(opt) {
        case 'c':
          options.config_file = optarg;
          break;
        case 'f':
          options.input_file = optarg;
          break;
        default:
          throw std::system_error(EINVAL, std::generic_category(), "invalid argument");
      }
    }
    if(optind  + 3 <= argc) {
      options.r1 = atoi(argv[optind++]);
      options.r2 = atoi(argv[optind++]);
      options.r3 = atoi(argv[optind++]);
    } else {
      throw std::system_error(EINVAL, std::generic_category(), "must pass dimentions");
    }

    return options;
  }


  std::vector<float> load_data_file(const char* path) {
    std::ifstream file(path);
    if(file) {
      std::vector<float> data;
      std::copy(
          std::istream_iterator<float>(file),
          std::istream_iterator<float>(),
          std::back_inserter(data)
          );
      return data;
    } else {
      throw std::system_error(ENOENT, std::generic_category(), "inputfile files does not exist");
    }
  }

}

int main(int argc, char *argv[])
{
  try{
    auto options = cmdline::parse_options(argc, argv);
    auto data = cmdline::load_data_file(options.input_file);
    sz_opencl_state* gpu_state;

    if(SZ_Init(options.config_file) == SZ_NSCS) {
      throw std::system_error(ENOENT, std::generic_category(), "config file does not exist");
    }
    if(sz_opencl_init(&gpu_state) == SZ_NSCS) {
      throw std::runtime_error(std::string(sz_opencl_error_msg(gpu_state)));
    }

    size_t out_size = 0;
    auto compression_start = chrono::system_clock::now();
    auto bytes = sz_compress_float3d_opencl(gpu_state, data.data(), options.r1, options.r2, options.r3, .99, &out_size);
    auto compression_end = chrono::system_clock::now();


    float* new_data;
    auto decompression_start = chrono::system_clock::now();
    sz_decompress_float_opencl(gpu_state, &new_data,
        /*r5*/0,/*r4*/0,/*r3*/options.r3,/*r2*/options.r2,/*r1*/options.r1,
        /*s5*/0,/*s4*/0,/*s3*/0,         /*s2*/0,         /*s1*/0, /*start_positions*/
        /*e5*/0,/*e4*/0,/*s3*/options.r3,/*e2*/options.r2,/*e1*/options.r1, /*end positions*/
        bytes,out_size);
    auto decompression_end = chrono::system_clock::now();

    auto compression_ms = chrono::duration_cast<chrono::milliseconds>(compression_end-compression_start).count();
    auto decompression_ms = chrono::duration_cast<chrono::milliseconds>(decompression_end-decompression_start).count();

    std::cout << compression_ms << std::endl;
    std::cout << decompression_ms << std::endl;

    sz_opencl_release(&gpu_state);
    SZ_Finalize();

  } catch (std::system_error const& e) {
    std::cerr << e.what() << ": " << e.code().message() << std::endl; 
    exit(EXIT_FAILURE);
  } catch (std::exception const& e) {
    std::cerr << e.what() << std::endl; 
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
