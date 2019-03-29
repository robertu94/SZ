#include <algorithm>
#include <tuple>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>
#include <unistd.h>
#include <iomanip>

namespace chrono = std::chrono;

#include <sz.h>

namespace cmdline {
  struct options {
    size_t r3=0, r2=0, r1=0;
    char const * input_file = nullptr;
    char const * config_file = nullptr;
		bool speedup = false;
  };

  const char* usage() {
    return R"(
./sz_gpu -c <sz.config> -f <input.f32>  r1 r2 r3
-c config file
-f input file
r1,r2,r3 dimentions of file to compression/decompres
)"; }

  options parse_options(int argc, char* argv[])
  {
    options options;
    int opt;
    while((opt = getopt(argc, argv, "sc:f:")) != -1)
    {
      switch(opt) {
        case 'c':
          options.config_file = optarg;
          break;
        case 'f':
          options.input_file = optarg;
          break;
				case 's':
					options.speedup = true;
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

	std::string
	basename(std::string const& filepath) {
		auto const pos = filepath.find_last_of('/');
		auto base = filepath.substr(pos+1);
		if (base.size() == 0)
			return filepath;
		else 
			return base;
	}

  std::vector<float> load_data_file(const char* path, size_t num_floats) {
    std::ifstream file(path, std::ios::binary);
    if(file) {
			std::vector<float> data(num_floats);
			file.read(reinterpret_cast<char*>(data.data()), num_floats * sizeof(float));
      return data;
    } else {
      throw std::system_error(ENOENT, std::generic_category(), "inputfile files does not exist");
    }
  }

}

template <class Compressor, class Decompressor>
std::tuple<double,double,double> evaluate(Compressor compressor, Decompressor decompresor, cmdline::options const& options, std::string const& method)
{
    auto data = cmdline::load_data_file(options.input_file, options.r1 * options.r2 * options.r3);

		auto inital_size = data.size() * sizeof(float);
    size_t compressed_size = 0;

    auto compression_start = chrono::system_clock::now();
    auto bytes = compressor(data.data(), options.r1, options.r2, options.r3, .99, &compressed_size);
    auto compression_end = chrono::system_clock::now();


    float* new_data;
    auto decompression_start = chrono::system_clock::now();
    decompresor(&new_data,
        /*r5*/0,/*r4*/0,/*r3*/options.r3,/*r2*/options.r2,/*r1*/options.r1,
        /*s5*/0,/*s4*/0,/*s3*/0,         /*s2*/0,         /*s1*/0, /*start_positions*/
        /*e5*/0,/*e4*/0,/*s3*/options.r3,/*e2*/options.r2,/*e1*/options.r1, /*end positions*/
        bytes, compressed_size);
    auto decompression_end = chrono::system_clock::now();

		//free(bytes);
		//free(new_data);

    auto compression_sec = chrono::duration_cast<chrono::milliseconds>(compression_end-compression_start).count()/1000.0;
    auto decompression_sec = chrono::duration_cast<chrono::milliseconds>(decompression_end-decompression_start).count()/1000.0;

		auto bytes_to_gb = 1024.0*1024.0*1024.0;
		auto reduction_ratio = inital_size/static_cast<double>(compressed_size);
		auto reduction_rate = (inital_size / (bytes_to_gb))/compression_sec;
		auto reconstruction_rate = (inital_size / (bytes_to_gb))/decompression_sec;

		std::cout << std::fixed << std::setprecision(6) << reduction_ratio << "," << reduction_rate << "," << reconstruction_rate << "," << method << "," << cmdline::basename(options.input_file) << std::endl;
		return {reduction_ratio, reduction_rate, reconstruction_rate};
}


int main(int argc, char *argv[])
{
  try{
    auto options = cmdline::parse_options(argc, argv);
    sz_opencl_state* gpu_state;

    if(SZ_Init(options.config_file) == SZ_NSCS) {
      throw std::system_error(ENOENT, std::generic_category(), "config file does not exist");
    }
    if(sz_opencl_init(&gpu_state) == SZ_NSCS) {
      throw std::runtime_error(std::string(sz_opencl_error_msg(gpu_state)));
    }

		auto gpu_compress = [&gpu_state](auto... args){
			return sz_compress_float3d_opencl(gpu_state, args...);
		};
		auto gpu_decompress = [&gpu_state](auto... args){
			sz_decompress_float_opencl(gpu_state, args...);
		};
		auto result_gpu = evaluate(gpu_compress, gpu_decompress, options, "gpu");

    sz_opencl_release(&gpu_state);
    SZ_Finalize();
		
    if(SZ_Init(options.config_file) == SZ_NSCS) {
      throw std::system_error(ENOENT, std::generic_category(), "config file does not exist");
    }

		auto cpu_compress = SZ_compress_float_3D_MDQ_decompression_random_access_with_blocked_regression; 
		auto cpu_decompress = SZ_decompress_args_randomaccess_float;
		auto result_cpu = evaluate(cpu_compress, cpu_decompress, options, "cpu");

    SZ_Finalize();

		if(options.speedup)
		{
			std::cout << "speedup compression" <<  std::get<1>(result_gpu)/std::get<1>(result_cpu) << std::endl;
			std::cout << "speedup decompression" <<  std::get<2>(result_gpu)/std::get<2>(result_cpu) << std::endl;
		}



  } catch (std::system_error const& e) {
		std::cerr << cmdline::usage() << std::endl;
    std::cerr << e.what() << std::endl; 
    exit(EXIT_FAILURE);
  } catch (std::exception const& e) {
    std::cerr << e.what() << std::endl; 
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
