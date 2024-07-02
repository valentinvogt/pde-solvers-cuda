#include "run_from_netcdf.hpp"

int main(int argc, char **argv) {
  std::string filename;
  if (argc == 1) {
    std::cout << "input filename to read: ";
    std::cin >> filename;
  } else {
    filename = argv[1];
  }

  zisa::device_type memory_location = zisa::device_type::cpu;
#if CUDA_AVAILABLE
  if (argc > 2 && strcmp(argv[2], "1") == 0) {
    memory_location = zisa::device_type::cuda;
  }
#endif

  NetCDFPDEReader reader(filename);
  int scalar_type = reader.get_scalar_type();

  auto glob_start = NOW;
  if (scalar_type == 0) {
    run_simulation<float>(reader, memory_location);
  } else if (scalar_type == 1) {
    run_simulation<double>(reader, memory_location);
  } else {
    std::cout << "Only double or float (scalar_type {0,1}) allowed"
              << std::endl;
    return -1;
  }
  auto glob_end = NOW;
  // std::cout << "duration of whole algorithm: " << DURATION(glob_end -
  // glob_start) << " ms" << std::endl;
  std::cout << DURATION(glob_end - glob_start) << std::endl;

  return 0;
}
#undef DURATION
