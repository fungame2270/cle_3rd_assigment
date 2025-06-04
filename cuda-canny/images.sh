#!/bin/bash

images_dir="./images"
runs=10
output_file="image_avg_times.csv"

# Write CSV header
echo "image_name,host_avg_ms,device_avg_ms" > "$output_file"

for image_path in "$images_dir"/*; do
  image_name=$(basename "$image_path")

  host_sum=0
  device_sum=0

  echo "Processing $image_name ..."

  for run in $(seq 1 $runs); do
    output=$(./canny "$image_path")

    host_time=$(echo "$output" | grep "Host processing time" | awk '{print $4}')
    device_time=$(echo "$output" | grep "Device processing time" | awk '{print $4}')

    host_sum=$(echo "$host_sum + $host_time" | bc)
    device_sum=$(echo "$device_sum + $device_time" | bc)
  done

  host_avg=$(echo "scale=6; $host_sum / $runs" | bc)
  device_avg=$(echo "scale=6; $device_sum / $runs" | bc)

  echo "$image_name,$host_avg,$device_avg" >> "$output_file"
done

echo "Done! Average times saved to $output_file"
