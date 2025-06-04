#!/bin/bash

output_file="canny_times_stats.csv"

# Write CSV header
echo "block_size,host_min_ms,host_max_ms,host_avg_ms,device_min_ms,device_max_ms,device_avg_ms" > "$output_file"

for block_size in $(seq 1 32); do
  host_times=()
  device_times=()

  echo "Testing block size $block_size ..."

  for run in $(seq 1 10); do
    output=$(./canny -b "$block_size")

    host_time=$(echo "$output" | grep "Host processing time" | awk '{print $4}')
    device_time=$(echo "$output" | grep "Device processing time" | awk '{print $4}')

    # Store times in arrays
    host_times+=("$host_time")
    device_times+=("$device_time")
  done

  # Function to compute min, max, avg from array
  function stats() {
    local arr=("${!1}")
    local min=${arr[0]}
    local max=${arr[0]}
    local sum=0

    for val in "${arr[@]}"; do
      (( $(echo "$val < $min" | bc -l) )) && min=$val
      (( $(echo "$val > $max" | bc -l) )) && max=$val
      sum=$(echo "$sum + $val" | bc)
    done
    local avg=$(echo "scale=6; $sum / ${#arr[@]}" | bc)
    echo "$min $max $avg"
  }

  host_stats=($(stats host_times[@]))
  device_stats=($(stats device_times[@]))

  echo "$block_size,${host_stats[0]},${host_stats[1]},${host_stats[2]},${device_stats[0]},${device_stats[1]},${device_stats[2]}" >> "$output_file"
done

echo "Done! Results saved to $output_file"
