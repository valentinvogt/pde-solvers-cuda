#ifndef CONVOLVE_SIGMA_ADD_F_CUDA_H_
#define CONVOLVE_SIGMA_ADD_F_CUDA_H_

#include <coupled_function.hpp>
#include <zisa/memory/array.hpp>

template <int n_coupled, typename Scalar, typename Function>
void convolve_sigma_add_f_cuda(zisa::array_view<Scalar, 2> dst,
                               zisa::array_const_view<Scalar, 2> src,
                               zisa::array_const_view<Scalar, 2> sigma,
                               Scalar del_x_2, Function f);

#define CALL_INSTANTIATION_N_COUPLED_5(TYPE)             \
    CALL_INSTANCIATION_MAX_POT_5(TYPE, 1)                \
    CALL_INSTANCIATION_MAX_POT_5(TYPE, 2)                \
    CALL_INSTANCIATION_MAX_POT_5(TYPE, 3)                \
    CALL_INSTANCIATION_MAX_POT_5(TYPE, 4)                \
    CALL_INSTANCIATION_MAX_POT_5(TYPE, 5)                

#define CALL_INSTANCIATION_MAX_POT_5(TYPE, N_COUPLED)            \
    PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA(TYPE, N_COUPLED, 1) \
    PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA(TYPE, N_COUPLED, 2) \
    PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA(TYPE, N_COUPLED, 3) \
    PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA(TYPE, N_COUPLED, 4) \
    PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA(TYPE, N_COUPLED, 5) 

#define PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA(TYPE, N_COUPLED, MAX_POT)   \
  extern template void convolve_sigma_add_f_cuda<N_COUPLED, TYPE, CoupledFunction<TYPE, N_COUPLED, MAX_POT>(                         \
      zisa::array_view<TYPE, 2>, zisa::array_const_view<TYPE, 2>,              \
      zisa::array_const_view<TYPE, 2>, TYPE, CoupledFunction<TYPE, N_COUPLED, MAX_POT>);

CALL_INSTANTIATION_N_COUPLED_5(float)
CALL_INSTANTIATION_N_COUPLED_5(double)

#undef CALL_INSTANCIATION_MAX_POT_5
#undef PDE_SOLVERS_CUDA_INSTANCIATE_CONVOLVE_SIGMA_CUDA
#undef CALL_INSTANTIATION_N_COUPLED_5


#endif
