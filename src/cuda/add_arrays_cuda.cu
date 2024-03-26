#include <cuda/add_arrays.hpp>
#include <cuda/add_arrays_impl.cuh>


template <typename Scalar>
void add_arrays_cuda(zisa::array_view<Scalar, 2> dst,
                     zisa::array_const_view<Scalar, 2> src, Scalar scaling);

#define PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(TYPE)                     \
  template void add_arrays_cuda<TYPE>(                                  \
      zisa::array_view<TYPE, 2>, zisa::array_const_view<TYPE, 2>, TYPE);

PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(float)
PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(double)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA
