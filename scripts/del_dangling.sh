#!/bin/bash

# Specify the directory
DIR="/cluster/scratch/vogtva/data/bruss/blowup"

# Iterate over all .json files
for json_file in "$DIR"/*.json; do
  # Strip the extension to get the base name
  base_name="${json_file%.json}"
  
  # Check if the corresponding _output.nc file exists
  if [[ ! -f "${base_name}_output.nc" ]]; then
    # Delete the .nc file if no _output.nc file exists
    rm "${base_name}.nc" echo "Deleted: ${base_name}.nc"
    rm "$json_file" && echo "Deleted: $json_file"
  fi
done