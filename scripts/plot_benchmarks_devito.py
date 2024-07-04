

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data_from_file(file_path, scale=False):
    data = {'Grid Size': [], 'Duration': []}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        grid_size = None
        times = []
        for line in lines:
            line = line.strip()
            if line.startswith("Grid size:"):
                if grid_size is not None:
                    data['Grid Size'].extend([grid_size] * len(times))
                    data['Duration'].extend(times)
                grid_size = int(line.split(":")[1].strip())
                times = []
            elif line.isdigit():
                # own implementation
                times.append(int(line))
            elif line.replace(".", "").isnumeric():
                # devito
                times.append(float(line) * 1000)
                

        if grid_size is not None:
            data['Grid Size'].extend([grid_size] * len(times))
            data['Duration'].extend(times)
    return pd.DataFrame(data)

def plot_loglog(data, label):
    avg_duration = data.groupby('Grid Size').mean()
    std_duration = data.groupby('Grid Size').std()
    
    plt.errorbar(avg_duration.index, avg_duration['Duration'], yerr=std_duration['Duration'], fmt='o-', label=label)

data_own_gpu = read_data_from_file('out/own_devito_gpu.txt')
data_own_cpu = read_data_from_file('out/own_devito_cpu.txt')
data_dev_cpu = read_data_from_file('out/out_cpu.txt')
data_dev_gpu = read_data_from_file('out/out_gpu.txt')

plt.figure(figsize=(12, 6))

plot_loglog(data_own_gpu, label="GPU own implementation")
plot_loglog(data_own_cpu, label="CPU own implementation")
plot_loglog(data_dev_gpu, label="GPU Devito implementation")
plot_loglog(data_dev_cpu, label="CPU Devito implementation")

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Grid Size')
plt.ylabel('Duration (ms)')
plt.title('Brusselator Algorithm Duration vs Grid Size')
plt.xticks([8, 16, 32, 64, 128, 256, 512, 1024, 2048], [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("out/bench_devito.pdf", bbox_inches='tight')

