#ifndef GENERIC_FUNCTION_HPP_
#define GENERIC_FUNCTION_HPP_

#if CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

#include <cmath>
template <typename Scalar> class GenericFunction {
public:
  GenericFunction() {}
  GenericFunction(Scalar const_val, Scalar lin_val, Scalar quad_val,
                  Scalar exp_scale_val, Scalar exp_pot_val)
      : const_val_(const_val), lin_val_(lin_val), quad_val_(quad_val),
        exp_scale_val_(exp_scale_val), exp_pot_val_(exp_pot_val) {}
#if CUDA_AVAILABLE
  inline __host__ __device__ Scalar operator()(Scalar x) { return const_val_ + lin_val_ * x + quad_val_ * x * x + exp_scale_val_ * std::exp(exp_pot_val_ * x); }
#else
  inline Scalar operator()(Scalar x) { return const_val_ + lin_val_ * x + quad_val_ * x * x + exp_scale_val_ * std::exp(exp_pot_val_ * x); }
#endif

  // TODO: setters
private:
  // functions returns f(x) = const_val_ + lin_val_ * x + quad_val_ * x^2 +
  // exp_scale_val_ * exp(exp_pot_val_ * x)
  Scalar const_val_ = 0.;
  Scalar lin_val_ = 0.;
  Scalar quad_val_ = 0.;
  Scalar exp_scale_val_ = 0.;
  Scalar exp_pot_val_ = 1.;
};

#endif // GENERIC_FUNCTION_HPP_
