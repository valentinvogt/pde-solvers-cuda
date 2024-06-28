#!/bin/bash

# Array of grid sizes to benchmark
SIZES=(32 64 128 256 512 1024)

# Number of repetitions for each size
REPETITIONS=4

# File to save the benchmarking results
RESULTS_FILE="out/benchmark_results.txt"

# Clear the results file if it exists
> $RESULTS_FILE

# Iterate over each grid size
for SIZE in "${SIZES[@]}"; do
    echo "Benchmarking for grid size: $SIZE"
    echo "Grid size: $SIZE" >> $RESULTS_FILE
    
    python scripts/benchmark_brusselator.py $SIZE
    # Repeat the benchmark for the given number of repetitions
    for ((i=1; i<=REPETITIONS; i++)); do
        echo "  Run $i for size $SIZE"

        # Generate the parameters using the Python script
        
        # Run the solver and capture the output
        DURATION=$(./build/run_from_netcdf data/benchmark.nc)
        
        # Append the duration to the results file
        echo "  Run $i duration: $DURATION" >> $RESULTS_FILE
    done

    echo "" >> $RESULTS_FILE
done

echo "Benchmarking completed. Results saved to $RESULTS_FILE"
