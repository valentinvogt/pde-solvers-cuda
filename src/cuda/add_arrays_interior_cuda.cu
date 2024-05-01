#include <cuda/add_arrays_interior_cuda.hpp>
#include <cuda/add_arrays_interior_cuda_impl.cuh>

template <int n_coupled, typename Scalar>
void add_arrays_interior_cuda(zisa::array_view<Scalar, 2> dst,
                              zisa::array_const_view<Scalar, 2> src,
                              Scalar scaling);

#define PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(N_COUPLED, TYPE)          \
  template void add_arrays_interior_cuda<N_COUPLED, TYPE>(                     \
      zisa::array_view<TYPE, 2>, zisa::array_const_view<TYPE, 2>, TYPE);

PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(1, float)
PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(1, double)
PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(2, float)
PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(2, double)
PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(3, float)
PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(3, double)
PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(4, float)
PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(4, double)
PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(5, float)
PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA(5, double)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_ADD_ARRAYS_CUDA
