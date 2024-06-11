#include <coupled_function_2.hpp>
#include <cuda/convolve_sigma_add_f_cuda.hpp>
#include <cuda/convolve_sigma_add_f_cuda_impl.cuh>

template <int n_coupled, typename Scalar, typename Function>
void convolve_sigma_add_f_cuda(zisa::array_view<Scalar, 2> dst,
                               zisa::array_const_view<Scalar, 2> src,
                               zisa::array_const_view<Scalar, 2> sigma,
                               Scalar del_x_2, Function f);

#define PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA_2(TYPE, N_COUPLED)    \
  template void                                                                \
      convolve_sigma_add_f_cuda<N_COUPLED, TYPE, CoupledFunction2<TYPE>>(      \
          zisa::array_view<TYPE, 2>, zisa::array_const_view<TYPE, 2>,          \
          zisa::array_const_view<TYPE, 2>, TYPE, CoupledFunction2<TYPE>);

PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA_2(float, 1)
PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA_2(double, 1)
PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA_2(float, 2)
PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA_2(double, 2)
PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA_2(float, 3)
PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA_2(double, 3)
PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA_2(float, 4)
PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA_2(double, 4)
PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA_2(float, 5)
PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA_2(double, 5)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA_2
