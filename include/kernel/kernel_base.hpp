#ifndef KERNEL_BASE_HPP_
#define KERNEL_BASE_HPP_

#include "zisa/memory/device_type.hpp"
#include <iostream>

template <typename Scalar, int rows, int cols> class KernelBase {
public:
  constexpr KernelBase(zisa::device_type device_type)
      : kernel_{}, device_type_(device_type), rows_(rows), cols_(cols) {}

  constexpr KernelBase(const KernelBase<Scalar, rows, cols> &other) :kernel_{}, rows_(rows), cols_(cols){
  }

  constexpr zisa::device_type memory_location() const { return this->device_type_; }

  constexpr int get_rows() const {
    return this->rows_;
  }
  constexpr int get_cols() const {
    return this->rows_;
  }

  constexpr unsigned shape(int dir) const {
    if (dir == 0) {
      return get_rows();
    } else if(dir == 1) {
      return get_rows();
    } else {
      std::cout << "shape only implemented for dir=0 or dir=1" << std::endl;
      return -1;
    }

  }

  constexpr const Scalar& operator()(int row, int col) const {
    return kernel_[row][col];
  }

  
  void print() const {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        std::cout << this->kernel_[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }



protected:
  const Scalar kernel_[rows][cols];
  const zisa::device_type device_type_ = zisa::device_type::unknown;
  const int rows_;
  const int cols_;
};

#endif // KERNEL_BASE_HPP_
