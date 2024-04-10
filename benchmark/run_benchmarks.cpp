#include "pde_base.hpp"
#include "zisa/memory/device_type.hpp"
#include <chrono>
#include <generic_function.hpp>
#include <pde_heat.hpp>
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

int main() {
  const int array_size_0 = 10;
  for (int i = 0; i < 5; i++) {
    const int array_size = array_size_0 * std::pow(2, i);
    zisa::array<float, 2> zero_values_cpu = create_value_data<float>(
        array_size, array_size, 0., zisa::device_type::cpu);
    zisa::array<float, 2> sigma_values_cpu = create_value_data<float>(
        2 * array_size - 3, array_size - 1, 0., zisa::device_type::cpu);

    GenericFunction<float> func;
    PDEHeat<float, GenericFunction<float>> pde_cpu(
        array_size - 2, array_size - 2, zisa::device_type::cpu,
        BoundaryCondition::Dirichlet, func, 1. / array_size, 1. / array_size);
    pde_cpu.read_values(zero_values_cpu.const_view(),
                        sigma_values_cpu.const_view(),
                        zero_values_cpu.const_view());
    // TODO: measure time and add cuda stuff
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
      pde_cpu.apply(0.1);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << array_size << "\t"
              << std::chrono::duration_cast<std::chrono::microseconds>(stop -
                                                                       start)
                     .count()
              << std::endl;
#if CUDA_AVAILABLE
    zisa::array<float, 2> zero_values_cuda = create_value_data<float>(
        array_size, array_size, 0., zisa::device_type::cuda);
    zisa::array<float, 2> sigma_values_cuda = create_value_data<float>(
        2 * array_size - 3, array_size - 1, 0., zisa::device_type::cuda);

    PDEHeat<float, GenericFunction<float>> pde_cuda(
        array_size - 2, array_size - 2, zisa::device_type::cuda,
        BoundaryCondition::Dirichlet, func, 1. / array_size, 1. / array_size);
    pde_cpu.read_values(zero_values_cpu.const_view(),
                        sigma_values_cpu.const_view(),
                        zero_values_cpu.const_view());
    // TODO: measure time and add cuda stuff
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
      pde_cuda.apply(0.1);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << array_size << "\t"
              << std::chrono::duration_cast<std::chrono::microseconds>(stop -
                                                                       start)
                     .count()
              << std::endl;
#endif
  }

  return 0;
}
