#ifndef NEUMANN_BC_CUDA_H_
#define NEUMANN_BC_CUDA_H_

#include <zisa/memory/array.hpp>

#ifndef THREAD_DIMS
#define THREAD_DIMS 1024
#endif

template <typename Scalar>
void neumann_bc_cuda(zisa::array<Scalar, 2> &data, unsigned n_ghost_cells_x,
                     unsigned n_ghost_cells_y);

#define PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(TYPE)  \
  extern template void neumann_bc_cuda<TYPE>(                 \
    zisa::array<TYPE, 2> &, unsigned, unsigned);            \

PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(float)
PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA(double)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_NEUMANN_BC_CUDA
#endif // NEUMANN_BC_CUDA_H_
