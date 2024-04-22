#include <cuda/periodic_bc_cuda.hpp>
#include <cuda/periodic_bc_cuda_impl.cuh>

template <typename Scalar, int n_coupled>
void periodic_bc_cuda(zisa::array_view<Scalar, 2> data);

#define PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA(TYPE, N_COUPLED)         \
  template void periodic_bc_cuda<TYPE, N_COUPLED>(zisa::array_view<TYPE, 2>);

PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA(float, 1)
PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA(double, 1)
PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA(float, 2)
PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA(double, 2)
PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA(float, 3)
PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA(double, 3)

#undef PDE_SOLVERS_CUDA_INSTANCIATE_PERIODIC_BC_CUDA
