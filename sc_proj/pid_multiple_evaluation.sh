#!/bin/bash

dataset="datlinger"
metric="semi_rigorous_Edistance"

for pid_percent in 1 10 20 30 40 50
do
  pid_ratio=$(printf "%.2f" "$(echo "$pid_percent / 100" | bc -l)")

  for i in {1..5}
  do
    echo "Running $dataset with pid_percent=$pid_percent (pid_ratio=$pid_ratio), run $i"
    
    python evaluation.py \
      --models noperturb cae cpa scPRAM \
      --datasets ${dataset} \
      --config_path ./configs \
      --base_path ../outputs \
      --setting pid \
      --save_folder pid/pid${pid_percent}_${i} \
      --evaluation_metrics ${metric} \
      --pid_percentage $pid_ratio &
  done

  wait
done

echo "All runs completed."