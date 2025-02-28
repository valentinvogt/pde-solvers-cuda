import pytest
import numpy as np
import netCDF4 as nc
import os
from scripts.make_inputs import create_input_file, run_wrapper
from scripts.initial_conditions import InitialCondition, ModelParams, SimParams, RunInfo

def test_create_input_file():
    # Test parameters
    input_filename = "test/data/test_input.nc"
    output_filename = "test/data/test_output.nc"
    os.makedirs("test/data", exist_ok=True)
    
    # Create test input file
    create_input_file(
        input_filename,
        output_filename,
        type_of_equation=2,  # Reaction-diffusion
        x_size=32,
        x_length=1.0,
        y_size=32,
        y_length=1.0,
        boundary_value_type=2,  # Periodic
        scalar_type=0,  # float
        n_coupled=2,
        coupled_function_order=3,
        number_timesteps=100,
        final_time=1.0,
        number_snapshots=10,
        n_members=1,
        Du=0.2,
        Dv=0.1
    )
    
    # Verify file was created and has correct attributes
    assert os.path.exists(input_filename)
    
    with nc.Dataset(input_filename, 'r') as root:
        # Check global attributes
        assert root.type_of_equation == 2
        assert root.n_members == 1
        assert root.x_size == 32
        assert root.y_size == 32
        assert root.x_length == 1.0
        assert root.y_length == 1.0
        assert root.boundary_value_type == 2
        assert root.scalar_type == 0
        assert root.n_coupled == 2
        assert root.coupled_function_order == 3
        assert root.number_timesteps == 100
        assert root.final_time == 1.0
        assert root.number_snapshots == 10
        
        # Check variables exist
        assert 'initial_data' in root.variables
        assert 'sigma_values' in root.variables
        assert 'function_scalings' in root.variables
        assert 'Du' in root.variables
        assert 'Dv' in root.variables
        
        # Check variable shapes
        assert root.variables['initial_data'].shape == (1, 34, 68)  # (n_members, x_size+2, n_coupled*(y_size+2))
        assert root.variables['sigma_values'].shape == (1, 65, 33)  # (n_members, 2*x_size+1, y_size+1)
        assert root.variables['function_scalings'].shape == (1, 18)  # (n_members, n_coupled*(coupled_function_order**n_coupled))
        
        # Check diffusion coefficients
        assert root.variables['Du'][()] == 0.2
        assert root.variables['Dv'][()] == 0.1

def test_run_wrapper():
    # Test parameters
    model_params = ModelParams(A=0.5, B=0.1, Du=0.2, Dv=0.1)
    sim_params = SimParams(Nx=32, dx=0.03125, Nt=100, dt=0.01, n_snapshots=10)
    initial_condition = InitialCondition(
        type="random_uniform",
        u_min=0.0,
        u_max=1.0,
        v_min=0.0,
        v_max=1.0
    )
    run_info = RunInfo(model="fhn", run_id="test")
    filename = "test/data/test_run.nc"
    
    # Run the wrapper
    run_wrapper(
        model_params,
        sim_params,
        initial_condition,
        run_info,
        filename,
        random_seed=42
    )
    
    # Verify output files were created
    assert os.path.exists(filename)
    assert os.path.exists(filename.replace(".nc", ".json"))
    
    # Check contents of netCDF file
    with nc.Dataset(filename, 'r') as root:
        assert root.type_of_equation == 2
        assert root.n_members == 1
        assert root.x_size == 32
        assert root.y_size == 32
        assert root.boundary_value_type == 2
        assert root.scalar_type == 0
        assert root.n_coupled == 2
        assert root.number_timesteps == 100
        assert root.number_snapshots == 10
        assert root.variables['Du'][()] == 0.2
        assert root.variables['Dv'][()] == 0.1

def test_cleanup():
    # Clean up test files
    test_files = [
        "test/data/test_input.nc",
        "test/data/test_output.nc",
        "test/data/test_run.nc",
        "test/data/test_run.json"
    ]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    os.rmdir("test/data") 