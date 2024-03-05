#include <cuda/convolve_cuda.hpp>
#include <cuda/convolve_cuda_impl.cuh>

template <typename Scalar>
void convolve_cuda(zisa::array_view<Scalar, 2> dst,
                   const zisa::array_const_view<Scalar, 2> &src,
                   const zisa::array_const_view<Scalar, 2> &kernel);

#define PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_CUDA(TYPE)                       \
  template void convolve_cuda<TYPE>(zisa::array_view<TYPE, 2>,                 \
                                    const zisa::array_const_view<TYPE, 2> &,   \
                                    const zisa::array_const_view<TYPE, 2> &);

PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_CUDA(float)
PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_CUDA(double)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_CUDA
