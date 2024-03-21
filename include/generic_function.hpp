#ifndef GENERIC_FUNCTION_HPP_
#define GENERIC_FUNCTION_HPP_

template <typename Scalar> class GenericFunction {
#if CUDA_AVAILABLE
  inline __host__ __device__ Scalar operator()(Scalar x) { return 1000; }
#else
  inline Scalar operator()(Scalar x) { return 1000; }
#endif
};

#endif // GENERIC_FUNCTION_HPP_
