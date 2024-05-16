'''
TODO: size of sigma values don't match and make shure that sizes of x and y matches everywhere (do not include boundary)

This file should be an easy template to generate netcdf input files for the pde-solvers-cuda function.
You can also create this files by yourself with another method but the file has to have following structure:
NetCDFFile {
    Attributes {
        type_of_equation-> int (0=HeatEquation, 1=WaveEquation)
        x_size -> int (number of nodes OF ONE PDE in x-direction WITHOUT boundary condition nodes)
        y_size -> int (number of nodes OF ONE PDE in y-direction WITHOUT boundary condition nodes)
        x_length -> Scalar (length of grid in x-direction)
        y_length -> Scalar (length of grid in y-direction)
        boundary_value_type -> int (0=Dirichlet, 1=Neumann, 2=Periodic)
        scalar_type -> int (0=float, 1=double)
        n_coupled -> int (#number coupled PDEs)
        coupled_function_order -> int (order of coupled function, 
                                     example: n_coupled=2, coupled_function_order=2 => f(x, y) = a + bx + cy + dxy)
        number_timesteps -> int (number of timesteps of simulation)
        final_time -> float (final time after simulation)
        number_snapshots -> int (number snapshots to shoot, includes initial time values)
        file_to_save_output -> string (file to save output)
    }
    Variables {
        initial_data -> scalar_type^(x_size + 2, n_coupled * (y_size + 2))
        sigma_values -> scalar_type^(2*x_size - 3, y_size - 1)
        [bc_neumann_values -> scalar_type^(x_size + 2, n_coupled * (y_size + 2))] (only if Boundary_value_type=1)
        function_scalings -> scalar_type^(n_coupled * coupled_function_order)
    }
}

The data of sigma values is arranged in following way:


   x_00              x_01              x_02              x_03
                       |                 |
                     sigma_00          sigma_01
                       |                 |
   x_10 - sigma_10 - x_11 - sigma_11 - x_12 - sigma_12 - x_13
                       |                 |
                     sigma_20          sigma_21
                       |                 |
   x_20              x_21              x_22              x_23

                                ||
                                \/

                sigma_00, sigma_01, 0
                sigma_10, sigma_11, sigma_12
                sigma_20, sigma_21, 0

'''
import netCDF4 as nc
import numpy as np
    
def create_input_file(filename, file_to_save_output, type_of_equation=0, 
                      x_size=8, y_size=8, x_length=1., y_length=1., boundary_value_type=1,
                      scalar_type=0, n_coupled=1, 
                      coupled_function_order=2, number_timesteps=1000,
                      final_time=1., number_snapshots=3):
    # Create a new NetCDF file
    with nc.Dataset(filename, 'w') as root:
        # Define attributes
        root.type_of_equation = type_of_equation  # 0=HeatEquation, 1=WaveEquation
        root.x_size = x_size  # Number of nodes OF ONE PDE in x-direction WITHOUT boundary condition nodes
        root.y_size = y_size  # Number of nodes OF ONE PDE in y-direction WITHOUT boundary condition nodes
        root.x_length = x_length # length of grid in x-direction
        root.y_length = y_length # length of grid in y-direction
        root.boundary_value_type = boundary_value_type  # 0=Dirichlet, 1=Neumann, 2=Periodic
        root.scalar_type = scalar_type  # 0=float, 1=double
        root.n_coupled = n_coupled  # Number of coupled PDEs
        root.coupled_function_order = coupled_function_order  # Order of coupled function
        root.number_timesteps = number_timesteps # number of timesteps of simulation
        root.final_time = final_time # final time after simulation
        root.number_snapshots = number_snapshots # number snapshots to shoot, includes initial time values
        root.file_to_save_output = file_to_save_output


        # Define variables
        if scalar_type == 0:
            scalar_type_string = 'f4' # float
        elif scalar_type == 1:
            scalar_type_string = 'f8' # double
        else:
            print("error: only scalar_type 0 or 1 allowed!")
            exit(-1)

        x_size = root.createDimension("x_size", root.x_size + 2)
        y_size = root.createDimension("y_size", root.n_coupled * (root.y_size + 2))
        x_size = root.createDimension("x_size_sigma", 2 * root.x_size - 3)
        y_size = root.createDimension("y_size_sigma", root.y_size - 1)
        function_scalings_x = root.createDimension("function_scaling_size", root.n_coupled * root.coupled_function_order)

        initial_data = root.createVariable('initial_data', scalar_type_string, ('x_size', 'y_size'))
        initial_data[:, :] = 1.
        sigma_values = root.createVariable('sigma_values', scalar_type_string, ('x_size_sigma', 'y_size_sigma'))
        sigma_values[:, :] = 2.
        function_scalings = root.createVariable('function_scalings', scalar_type_string, ("function_scaling_size"))
        function_scalings[:] = np.linspace(0., 1., root.n_coupled * root.coupled_function_order)

        # If boundary_value_type is Neumann, define additional variable
        if root.boundary_value_type == 1:
            bc_neumann_values = root.createVariable('bc_neumann_values', scalar_type_string, ('x_size', 'y_size'))
            bc_neumann_values[:, :] = 3.


    print(f"NetCDF file '{filename}' created successfully.")

# Usage example:
create_input_file('data/example.nc', 'data/example_out.nc')