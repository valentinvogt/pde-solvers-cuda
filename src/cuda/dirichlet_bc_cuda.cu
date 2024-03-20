#include <cuda/dirichlet_bc_cuda.hpp>
#include <cuda/dirichlet_bc_cuda_impl.cuh>

template <typename Scalar>
void dirichlet_bc_cuda(zisa::array_view<Scalar, 2> data,
                       zisa::array_const_view<Scalar, 2> bc);

#define PDE_SOLVERS_CUDA_INSTANCIATE_DIRICHLET_BC_CUDA(TYPE)                   \
  template void dirichlet_bc_cuda<TYPE>(zisa::array_view<TYPE, 2>,             \
                                        zisa::array_const_view<TYPE, 2>);

PDE_SOLVERS_CUDA_INSTANCIATE_DIRICHLET_BC_CUDA(float)
PDE_SOLVERS_CUDA_INSTANCIATE_DIRICHLET_BC_CUDA(double)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_DIRICHLET_BC_CUDA
