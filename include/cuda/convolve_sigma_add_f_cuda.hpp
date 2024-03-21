#ifndef CONVOLVE_SIGMA_ADD_F_CUDA_H_
#define CONVOLVE_SIGMA_ADD_F_CUDA_H_

#include <zisa/memory/array.hpp>
#include <generic_function.hpp>

template <typename Scalar, typename Function>
void convolve_sigma_add_f_cuda(zisa::array_view<Scalar, 2> dst,
                   zisa::array_const_view<Scalar, 2> src,
                   zisa::array_const_view<Scalar, 2> sigma, Scalar del_x_2, Function f);

#define PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA(TYPE, FUNC)                 \
  extern template void convolve_sigma_add_f_cuda<TYPE, FUNC>(zisa::array_view<TYPE, 2>,          \
                                           zisa::array_const_view<TYPE, 2>,    \
                                           zisa::array_const_view<TYPE, 2>, TYPE, FUNC);

PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA(float, GenericFunction<float>)
PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA(double, GenericFunction<double>)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA
#endif // CONVOLVE_SIGMA_ADD_F_CUDA_H_