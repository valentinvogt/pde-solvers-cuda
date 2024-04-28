#include <cuda/neumann_bc_cuda.hpp>
#include <cuda/neumann_bc_cuda_impl.cuh>

template <typename Scalar, int n_coupled>
void neumann_bc_cuda(zisa::array_view<Scalar, 2> data,
                     zisa::array_const_view<Scalar, 2> bc, Scalar dt);

#define PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(TYPE, N_COUPLED)          \
  template void neumann_bc_cuda<TYPE, N_COUPLED>(                              \
      zisa::array_view<TYPE, 2>, zisa::array_const_view<TYPE, 2>, TYPE);

PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(float, 1)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(double, 1)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(float, 2)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(double, 2)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(float, 3)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(double, 3)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA
