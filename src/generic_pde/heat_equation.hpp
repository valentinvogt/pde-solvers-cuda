#ifndef __HEAT_EQUATION__
#define __HEAT_EQUATION__

#include "pde.hpp"

/* Heat equation: (diff u) / (diff t) = laplacian u */
template<typename T>
class Heat_Equation: public PDE<T> {
public:
  using PDE<T>::PDE;

  // TODO: think about which bc are acceptable, accept some kind of bc_function?
  //       think about adding a bc class
  void add_boundary_condition() {
    this->_boundary_conditions_loaded = true;
  }

};

#endif //__HEAT_EQUATION__