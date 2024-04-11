import numpy as np
import matplotlib.pyplot as plt

# Read data from file
with open('data.txt', 'r') as file:
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
plt.xlabel('Array Size')
plt.ylabel('Time (milliseconds)')
plt.xticks(array_size)
plt.grid(True)
plt.legend()
plt.show()