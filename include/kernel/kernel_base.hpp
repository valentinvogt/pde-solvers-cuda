#ifndef KERNEL_BASE_HPP_
#define KERNEL_BASE_HPP_

#include "zisa/memory/device_type.hpp"

template <typename Scalar, int rows, int cols> class KernelBase {
public:
  constexpr zisa::device_type memory_location() const { return this->device_type_; }
  constexpr KernelBase(zisa::device_type device_type)
      : kernel_{}, device_type_(device_type), rows_(rows), cols_(cols) {}
  constexpr int get_rows() const {
    return this->rows_;
  }
  constexpr int get_cols() const {
    return this->rows_;
  }

  constexpr const Scalar& operator()(int row, int col) const {
    return kernel_[row][col];
  }
  


protected:
  const Scalar kernel_[rows][cols];
  const zisa::device_type device_type_ = zisa::device_type::unknown;
  const int rows_;
  const int cols_;
};

#endif // KERNEL_BASE_HPP_
