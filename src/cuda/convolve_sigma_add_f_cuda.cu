#include "generic_function.hpp"
#include <cuda/convolve_sigma_add_f_cuda.hpp>
#include <cuda/convolve_sigma_add_f_cuda_impl.cuh>

template <typename Scalar, typename Function>
void convolve_sigma_add_f_cuda(zisa::array_view<Scalar, 2> dst,
                               zisa::array_const_view<Scalar, 2> src,
                               zisa::array_const_view<Scalar, 2> sigma,
                               Scalar del_x_2, Function f);

#define PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA(TYPE, FUNC)           \
  template void convolve_sigma_add_f_cuda<TYPE, FUNC>(                         \
      zisa::array_view<TYPE, 2>, zisa::array_const_view<TYPE, 2>,              \
      zisa::array_const_view<TYPE, 2>, Scalar, FUNC);

PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA(float, GenericFunction<float>)
PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA(double,
                                                 GenericFunction<double>)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA
