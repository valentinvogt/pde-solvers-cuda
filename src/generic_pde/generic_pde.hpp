#ifndef __GENERIC_PDE_HPP__
#define __GENERIC_PDE_HPP__

#include "pde.hpp"

template <typename T>
class GenericPDESolver {
public:
  GenericPDESolver();

  GenericPDESolver(PDE<T> pde) : _pde(pde){}

  void load_pde(PDE<T> pde) {}

  void solve() {}

private:
  PDE<T> _pde;
};

#endif //__GENERIC_PDE_HPP__