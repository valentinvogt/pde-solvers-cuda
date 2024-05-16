import numpy as np
import matplotlib.pyplot as plt

# Read data from file
with open('out/res_benchmark_6.txt', 'r') as file:
    next(file)  # Skip header
    data = np.loadtxt(file, delimiter=',', skiprows=0)

array_size = data[:, 0]
time_cpu = data[:, 1]
time_cuda = data[:, 2]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(array_size, time_cpu, marker='o', label='CPU Time')
plt.plot(array_size, time_cuda, marker='o', label='CUDA Time')
plt.title('Execution Time Comparison (CPU vs CUDA)')
plt.xlabel('Array size')
plt.ylabel('Time (microseconds)')
stepsize = int(len(array_size) / 10.)
plt.xticks(array_size[::stepsize])
plt.grid(True)
plt.legend()
plt.yscale('log')
plt.xscale('log')


plt.savefig('out/res_benchmark_6.pdf')