#ifndef GENERIC_FUNCTION_HPP_
#define GENERIC_FUNCTION_HPP_

template <typename Scalar>
class GenericFunction {
#if CUDA_AVAILABLE
  virtual inline __host__ __device__ Scalar operator()(Scalar x) = 0;
#else
  virtual inline Scalar operator()(Scalar x) = 0;
#endif
};

#endif // GENERIC_FUNCTION_HPP_
