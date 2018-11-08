#!/usr/bin/env bash
max=0
while sleep 0.1
do
	str=$(nvidia-smi --query-gpu=memory.used,utilization.memory --format=csv)
	max_new=$(cut -d ',' -f 1 <<< $str | tail -n 1 | cut -d ' ' -f 1)
	(( $max_new > max )) && max=$max_new
	all_data=$(tail -n 1 <<< $str)
	mem_usg=$(free -m | sed '2q;d' | tr -s [:blank:]  | cut -d $' ' -f 4)
	clear;echo "used_vram: $all_data, max_vram: $max MiB, free_mem: $mem_usg MiB"
done
