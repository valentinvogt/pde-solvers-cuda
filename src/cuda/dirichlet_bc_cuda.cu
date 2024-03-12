#include <cuda/dirichlet_bc_cuda.hpp>
#include <cuda/dirichlet_bc_cuda_impl.cuh>

template <typename Scalar>
void dirichlet_bc_cuda(zisa::array<Scalar, 2> &data, unsigned n_ghost_cells_x,
                     unsigned n_ghost_cells_y, Scalar value);

#define PDE_SOLVERS_CUDA_INSTANCIATE_DIRICHLET_BC_CUDA(TYPE)  \
  template void dirichlet_bc_cuda<TYPE>(                 \
    zisa::array<TYPE, 2> &, unsigned, unsigned, TYPE);            \

PDE_SOLVERS_CUDA_INSTANCIATE_DIRICHLET_BC_CUDA(float)
PDE_SOLVERS_CUDA_INSTANCIATE_DIRICHLET_BC_CUDA(double)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_DIRICHLET_BC_CUDA
