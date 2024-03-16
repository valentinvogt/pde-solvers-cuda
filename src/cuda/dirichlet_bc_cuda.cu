#include <cuda/dirichlet_bc_cuda.hpp>
#include <cuda/dirichlet_bc_cuda_impl.cuh>

template <typename Scalar>
void dirichlet_bc_cuda(zisa::array_view<Scalar, 2> data,
                       const zisa::array_const_view<Scalar, 2> &bc,
                       unsigned n_ghost_cells_x,
                       unsigned n_ghost_cells_y);

#define PDE_SOLVERS_CUDA_INSTANCIATE_DIRICHLET_BC_CUDA(TYPE)  \
  template void dirichlet_bc_cuda<TYPE>(                 \
    zisa::array_view<TYPE, 2>, const zisa::array_const_view<TYPE, 2> &, unsigned, unsigned);            \

PDE_SOLVERS_CUDA_INSTANCIATE_DIRICHLET_BC_CUDA(float)
PDE_SOLVERS_CUDA_INSTANCIATE_DIRICHLET_BC_CUDA(double)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_DIRICHLET_BC_CUDA
