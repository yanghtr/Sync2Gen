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
    python vis_pred_sync.py --type $type --start $counter --length $interval --pred_dir ../dump/$type/$tmodel --sync_dir ./dump/$type/$tmodel/$exp_name  --vis_dir ./vis/$type/$tmodel/$exp_name  &
done


