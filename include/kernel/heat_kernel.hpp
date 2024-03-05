#ifndef HEAT_KERNEL_HPP_
#define HEAT_KERNEL_HPP_
#include "zisa/memory/device_type.hpp"
#include <kernel_base.hpp>

template <typename Scalar> class HeatKernel : public KernelBase<Scalar, 3, 3> {
public:
  constexpr HeatKernel(double k, double dt, zisa::device_type device_type)
      : KernelBase<Scalar, 3, 3>(device_type), k_(k), dt_(dt) {}

private:
  constexpr void initialize_kernel() {
    const Scalar scaling = k_ / (dt_ * dt_);
    // Initialize kernel_ values using constexpr logic
    this->kernel_[0][0] = 0.0;
    this->kernel_[0][1] = scaling;
    this->kernel_[0][2] = 0.0;
    this->kernel_[1][0] = scaling;
    this->kernel_[1][1] = 1.0 - 4.0 * scaling;
    this->kernel_[1][2] = scaling;
    this->kernel_[2][0] = 0.0;
    this->kernel_[2][1] = scaling;
    this->kernel_[2][2] = 0.0;
  }

  const Scalar dt_;
  const Scalar k_;
};

#endif // HEAT_KERNEL_HPP_
