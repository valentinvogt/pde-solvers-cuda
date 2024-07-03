#include "pde_base.hpp"
#include "zisa/memory/device_type.hpp"
#include <chrono>
#include <coupled_function.hpp>
#include <pde_heat.hpp>
#include <cstring>
// helper function which creates simple data array where all values are set to
// value, if CUDA_AVAILABLE on gpu, else on cpu
template <typename Scalar>
inline zisa::array<Scalar, 2>
create_value_data(int x_size, int y_size, Scalar value,
                  zisa::device_type memory_location) {
  zisa::array<Scalar, 2> data(zisa::shape_t<2>(x_size, y_size),
                              zisa::device_type::cpu);
  for (int i = 0; i < x_size; i++) {
    for (int j = 0; j < y_size; j++) {
      data(i, j) = value;
    }
  }
  if (memory_location == zisa::device_type::cpu) {
    return data;
  }
#if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    zisa::array<Scalar, 2> data_gpu(zisa::shape_t<2>(x_size, y_size),
                                    zisa::device_type::cuda);
    zisa::copy(data_gpu, data);
    return data_gpu;
  }
#endif
  else {
    std::cout << "device type not supported yet in test_heat_equation"
              << std::endl;
    exit(-1);
  }
}

void run_benchmark_gpu_vs_cpu_size(int n_timesteps) {
  const int array_size_0 = 10;
#if CUDA_AVAILABLE
  std::cout << "# array_size, time_cpu, time_gpu" << std::endl;
#else
  std::cout << "# array_size, time_cpu" << std::endl;
#endif
  for (int size = 1; size < 100; size++) {
    int n_iters = 5;
    double time_cpu = 0;
#if CUDA_AVAILABLE
    double time_cuda = 0;
#endif
    const int array_size = array_size_0 * size;
    for (int iters = 0; iters < n_iters + 1; iters++) {
      zisa::array<float, 2> zero_values_cpu = create_value_data<float>(
          array_size, array_size, 0., zisa::device_type::cpu);
      zisa::array<float, 2> sigma_values_cpu = create_value_data<float>(
          2 * array_size - 3, array_size - 1, 0., zisa::device_type::cpu);

      zisa::array<float, 1> function_scalings(zisa::shape_t<1>(1), zisa::device_type::cpu);
      function_scalings(0) = 0.;
      CoupledFunction<float> func(function_scalings.const_view(), 1, 1);
      PDEHeat<1, float, CoupledFunction<float>> pde_cpu(
          array_size - 2, array_size - 2, zisa::device_type::cpu,
          BoundaryCondition::Dirichlet, func, 1. / array_size, 1. / array_size);
      pde_cpu.read_values(zero_values_cpu.const_view(),
                          sigma_values_cpu.const_view(),
                          zero_values_cpu.const_view());
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < n_timesteps; i++) {
        pde_cpu.apply(0.1);
      }
      auto stop = std::chrono::high_resolution_clock::now();
      // do not measure the first one
      if (iters != 0) {
        time_cpu +=
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                .count();
      }
#if CUDA_AVAILABLE
      zisa::array<float, 2> zero_values_cuda = create_value_data<float>(
          array_size, array_size, 0., zisa::device_type::cuda);
      zisa::array<float, 2> sigma_values_cuda = create_value_data<float>(
          2 * array_size - 3, array_size - 1, 0., zisa::device_type::cuda);

      zisa::array<float, 1> function_scalings_cuda(zisa::shape_t<1>(1), zisa::device_type::cuda);
      zisa::copy(function_scalings_cuda, function_scalings);
      CoupledFunction<float> func_cuda(function_scalings_cuda.const_view(), 1, 1);
      PDEHeat<1, float, CoupledFunction<float>> pde_cuda(
          array_size - 2, array_size - 2, zisa::device_type::cuda,
          BoundaryCondition::Dirichlet, func_cuda, 1. / array_size, 1. / array_size);
      pde_cuda.read_values(zero_values_cpu.const_view(),
                           sigma_values_cpu.const_view(),
                           zero_values_cpu.const_view());
      start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < n_timesteps; i++) {
        pde_cuda.apply(0.1);
      }
      stop = std::chrono::high_resolution_clock::now();
      // do not measure the first one
      if (iters != 0) {
        time_cuda +=
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                .count();
      }
#endif
    }
    std::cout << array_size << "," << time_cpu / (double)n_iters
#if CUDA_AVAILABLE
                << "," << time_cuda / (double)n_iters
#endif
                << std::endl;
  }
}

void run_benchmark_gpu_vs_cpu_n_timesteps(int array_size) {
  const int n_timesteps_base = 10;
#if CUDA_AVAILABLE
  std::cout << "# n_timesteps, time_cpu, time_gpu" << std::endl;
#else
  std::cout << "# n_timesteps, time_cpu" << std::endl;
#endif
  for (int n_timesteps = 0; n_timesteps < 100; n_timesteps++) {
    int n_iters = 5;
    double time_cpu = 0;
#if CUDA_AVAILABLE
    double time_cuda = 0;
#endif
    for (int iters = 0; iters < n_iters + 1; iters++) {
      zisa::array<float, 2> zero_values_cpu = create_value_data<float>(
          array_size, array_size, 0., zisa::device_type::cpu);
      zisa::array<float, 2> sigma_values_cpu = create_value_data<float>(
          2 * array_size - 3, array_size - 1, 0., zisa::device_type::cpu);

      zisa::array<float, 1> function_scalings(zisa::shape_t<1>(1), zisa::device_type::cpu);
      function_scalings(0) = 0.;
      CoupledFunction<float> func(function_scalings.const_view(), 1,1 );
      PDEHeat<1, float, CoupledFunction<float>> pde_cpu(
          array_size - 2, array_size - 2, zisa::device_type::cpu,
          BoundaryCondition::Dirichlet, func, 1. / array_size, 1. / array_size);
      pde_cpu.read_values(zero_values_cpu.const_view(),
                          sigma_values_cpu.const_view(),
                          zero_values_cpu.const_view());
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < n_timesteps * n_timesteps_base; i++) {
        pde_cpu.apply(0.1);
      }
      auto stop = std::chrono::high_resolution_clock::now();
      // do not measure the first one
      if (iters != 0) {
        time_cpu +=
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                .count();
      }
#if CUDA_AVAILABLE
      zisa::array<float, 2> zero_values_cuda = create_value_data<float>(
          array_size, array_size, 0., zisa::device_type::cuda);
      zisa::array<float, 2> sigma_values_cuda = create_value_data<float>(
          2 * array_size - 3, array_size - 1, 0., zisa::device_type::cuda);

      zisa::array<float, 1> function_scalings_cuda(zisa::shape_t<1>(1), zisa::device_type::cuda);
      zisa::copy(function_scalings_cuda, function_scalings);
      CoupledFunction<float> func_cuda(function_scalings_cuda.const_view(),1 ,1 );
      PDEHeat<1, float, CoupledFunction<float>> pde_cuda(
          array_size - 2, array_size - 2, zisa::device_type::cuda,
          BoundaryCondition::Dirichlet, func_cuda, 1. / array_size, 1. / array_size);
      pde_cuda.read_values(zero_values_cpu.const_view(),
                           sigma_values_cpu.const_view(),
                           zero_values_cpu.const_view());
      start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < n_timesteps * n_timesteps_base; i++) {
        pde_cuda.apply(0.1);
      }
      stop = std::chrono::high_resolution_clock::now();
      // do not measure the first one
      if (iters != 0) {
        time_cuda +=
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                .count();
      }
#endif
    }
    std::cout << n_timesteps * n_timesteps_base << "," << time_cpu / (double)n_iters
#if CUDA_AVAILABLE
                << "," << time_cuda / (double)n_iters
#endif
                << std::endl;
  }
}

int main(int argc, char ** argv) {
  if (argc > 1) {
    if (!strcmp(argv[1], "gpu_cpu_size")) {
      int n_timesteps = 500;
      if (argc > 2) {
        n_timesteps = std::stoi(argv[2]);
      }
      run_benchmark_gpu_vs_cpu_size(n_timesteps);   
    } else if (!strcmp(argv[1], "gpu_cpu_timesteps")) {
      int array_size = 128;
      if (argc > 2) {
        array_size = std::stoi(argv[2]);
      }
      run_benchmark_gpu_vs_cpu_n_timesteps(array_size);   
    } else {
      std::cout << "usage: ./build/benchmarks {gpu_cpu_size {n_timesteps}, gpu_cpu_timesteps {array_size}}" << std::endl;
    }
  } else {
    std::cout << "usage: ./build/benchmarks {gpu_cpu_size {n_timesteps}, gpu_cpu_timesteps {array_size}}" << std::endl;
  }

  return 0;
}
