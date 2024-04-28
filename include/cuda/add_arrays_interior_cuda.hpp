#ifndef ADD_ARRAYS_CUDA_H_
#define ADD_ARRAYS_CUDA_H_

#include <zisa/memory/array.hpp>

template <typename Scalar>
void add_arrays_interior_cuda(zisa::array_view<Scalar, 2> dst,
                              zisa::array_const_view<Scalar, 2> src,
                              Scalar scaling);

#define PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(TYPE)                     \
  extern template void add_arrays_interior_cuda<TYPE>(                         \
      zisa::array_view<TYPE, 2>, zisa::array_const_view<TYPE, 2>, TYPE);

PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(float)
PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(double)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA

#endif // ADD_ARRAYS_CUDA_H_
