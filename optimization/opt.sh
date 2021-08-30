#!/bin/bash
ARGV=("$@")
interval=4
start=0
end=100
type=${ARGV[0]}
tmodel=${ARGV[1]}
exp_name=${ARGV[2]}

for (( counter=start; counter<end; counter+=interval  )); do
    echo $counter
    CUDA_VISIBLE_DEVICES="" python opt_final_joint.py --type $type --dump_results --dump_dir  ./dump/$type/$tmodel/$exp_name --data_dir ../dump/$type/$tmodel --log_dir ./log --start $counter --length $interval &
done

