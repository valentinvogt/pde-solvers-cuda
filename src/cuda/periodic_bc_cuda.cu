#include <cuda/periodic_bc_cuda.hpp>
#include <cuda/periodic_bc_cuda_impl.cuh>

#ifndef PERIODIC_BC_CUDA_H_
#define PERIODIC_BC_CUDA_H_

#include <zisa/memory/array.hpp>

template <typename Scalar>
void periodic_bc_cuda(zisa::array<Scalar, 2> &data, unsigned n_ghost_cells_x,
                     unsigned n_ghost_cells_y);

#define PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA(TYPE)  \
  template void periodic_bc_cuda<TYPE>(                 \
    zisa::array<TYPE, 2> &, unsigned, unsigned);            \

PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA(float)
PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA(double)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA
#endif // PERIODIC_BC_CUDA_H_
