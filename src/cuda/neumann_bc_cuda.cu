#include <cuda/neumann_bc_cuda.hpp>
#include <cuda/neumann_bc_cuda_impl.cuh>

template <typename Scalar>
void neumann_bc_cuda(zisa::array_view<Scalar, 2> &data, 
                     zisa::array_const_view<Scalar, 2> &bc, unsigned n_ghost_cells_x,
                     unsigned n_ghost_cells_y);

#define PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(TYPE)  \
  template void neumann_bc_cuda<TYPE>(                 \
    zisa::array_view<TYPE, 2> &, zisa::array_const_view<TYPE, 2> &, unsigned, unsigned);            \

PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(float)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(double)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA
