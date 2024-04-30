#include <cuda/neumann_bc_cuda.hpp>
#include <cuda/neumann_bc_cuda_impl.cuh>

template <int n_coupled, typename Scalar>
void neumann_bc_cuda(zisa::array_view<Scalar, 2> data,
                     zisa::array_const_view<Scalar, 2> bc, Scalar dt);

#define PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(N_COUPLED, TYPE)          \
  template void neumann_bc_cuda<N_COUPLED, TYPE>(                              \
      zisa::array_view<TYPE, 2>, zisa::array_const_view<TYPE, 2>, TYPE);

PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(1, float)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(1, double)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(2, float)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(2, double)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(3, float)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(3, double)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(4, float)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(4, double)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(5, float)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(5, double)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA
