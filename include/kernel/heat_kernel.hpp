#ifndef HEAT_KERNEL_HPP_
#define HEAT_KERNEL_HPP_
#include "zisa/memory/device_type.hpp"
#include <kernel_base.hpp>

template <typename Scalar> class HeatKernel : public KernelBase<Scalar, 3, 3> {
public:
  constexpr HeatKernel(double k, double dt, zisa::device_type device_type)
      : KernelBase<Scalar, 3, 3>(device_type), k_dt_dt_(k / (dt* dt)) {}

private:
  const Scalar k_dt_dt_;
  const Scalar kernel_[3][3] = {0, k_dt_dt_, 0,
                                k_dt_dt_,1 -  4. * k_dt_dt_, k_dt_dt_,
                                0, k_dt_dt_, 0};
};

#endif // HEAT_KERNEL_HPP_
