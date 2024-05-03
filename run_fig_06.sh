#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)*0.6" | bc`
num_threads=$(round ${num_threads} 0)
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

exe_2=build/examples/test_poisson

mesh=dat/sphere32.off
query_mesh=dat/query_plane_remesh.off

outdir=result/kl_res/supernodes/
sec_num=16

mkdir -p ${outdir}
${exe_2} ${mesh} ${outdir} ${sec_num} ${query_mesh}
