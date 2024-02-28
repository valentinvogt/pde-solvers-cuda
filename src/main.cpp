#include <iostream>
#include <pde_base.hpp>
#include <zisa/memory/array.hpp>

int main() {
  // Set up the finite difference kernel for the heat equation
  zisa::array<float, 2> heat_kernel(zisa::shape_t<2>(3, 3));
  float scaling = 0.1; // 1 / dt^2
  heat_kernel(0, 0) = 0;
  heat_kernel(0, 1) = 1 * scaling;
  heat_kernel(0, 2) = 0;
  heat_kernel(1, 0) = 1 * scaling;
  heat_kernel(1, 1) = -4 * scaling;
  heat_kernel(1, 2) = 1 * scaling;
  heat_kernel(2, 0) = 0;
  heat_kernel(2, 1) = 1 * scaling;
  heat_kernel(2, 2) = 0;
  // Construct a PDE based on the given kernel
  PDEBase<float> pde(128, 128, heat_kernel);
  // TODO: apply initial conditions
  // Do some things with it...
  pde.apply();
  // Do some more things...
  // TODO
  return 0;
}
