#include <cuda/periodic_bc_cuda.hpp>
#include <cuda/periodic_bc_cuda_impl.cuh>

template <typename Scalar>
void periodic_bc_cuda(zisa::array_view<Scalar, 2> data);

#define PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA(TYPE)                    \
  template void periodic_bc_cuda<TYPE>(zisa::array_view<TYPE, 2>);

PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA(float)
PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA(double)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA
