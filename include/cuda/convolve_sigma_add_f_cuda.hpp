#ifndef CONVOLVE_SIGMA_ADD_F_CUDA_H_
#define CONVOLVE_SIGMA_ADD_F_CUDA_H_

/*
TODO: function types?
#include <zisa/memory/array.hpp>

template <typename Scalar, typename Function>
void convolve_sigma_add_f_cuda(zisa::array_view<Scalar, 2> dst,
                   zisa::array_const_view<Scalar, 2> src,
                   zisa::array_const_view<Scalar, 2> kernel);

#define PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_CUDA(TYPE, FUNC)                 \
  extern template void convolve_cuda<TYPE>(zisa::array_view<TYPE, 2>,          \
                                           zisa::array_const_view<TYPE, 2>,    \
                                           zisa::array_const_view<TYPE, 2>);

PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_CUDA(float)
PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_CUDA(double)

*/
#undef PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_CUDA
#endif // CONVOLVE_SIGMA_ADD_F_CUDA_H_