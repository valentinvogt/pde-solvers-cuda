import netCDF4 as nc

def read_file(filename):

    with nc.Dataset(filename, 'r') as root:
        print(root)
        print(root.variables['initial_data'][:, :])



read_file('data/example.nc')
    
