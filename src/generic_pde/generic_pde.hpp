#ifndef __GENERIC_PDE_HPP__
#define __GENERIC_PDE_HPP__

#include <algorithm>

template <typename T> class GenericPDE {
public:

  GenericPDE(unsigned int x_dim, unsigned int y_dim)
      : _x_dim(x_dim), _y_dim(y_dim), _state_loaded(false),
        _boundary_conditions_loaded(false) {
    this->_current_state = new T[x_dim * y_dim];
  }

  ~GenericPDE() {
    if (this->_state_loaded) {
      delete[] this->_current_state;
    }
  }

  void set_initial_conditions(const T *initial_conditions) {
    std::copy(initial_conditions,
              initial_conditions + this->_x_dim * this->_y_dim,
              this->_current_state);
    this->_state_loaded = true;;
  }

  //TODO:
  virtual T *iterate_step(double dt) = 0;

protected:
  unsigned int _x_dim;
  unsigned int _y_dim;
  bool _state_loaded = false;
  bool _boundary_conditions_loaded = false;
  T *_current_state;
};

#endif //__GENERIC_PDE_HPP__