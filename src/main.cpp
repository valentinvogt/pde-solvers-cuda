#include "generic_pde/generic_pde.hpp"
#include "generic_pde/heat_equation.hpp"
#include <iostream>

int main() {
  Heat_Equation<float> heat_equation(/*x_dim*/ 10, /*y_dim*/ 10);
  heat_equation.add_boundary_condition();
  float init_conditions[100];

  heat_equation.set_initial_conditions(init_conditions);

  heat_equation.iterate_step(0.1);
}
