#!/bin/bash

dataset="datlinger"

for pid_percent in 1 10 20 30 40 50
do
  pid_ratio=$(printf "%.2f" "$(echo "$pid_percent / 100" | bc -l)")

  for i in {1..5}
  do
    echo "Running $dataset with pid_percent=$pid_percent (pid_ratio=$pid_ratio), run $i"

    python prediction.py \
      --models cae cpa \
      --datasets ${dataset} \
      --config_path ./configs \
      --setting pid \
      --save_folder pid/pid${pid_percent}_${i} \
      --pid_percentage $pid_ratio &
  done

  wait
done

echo "All runs completed."