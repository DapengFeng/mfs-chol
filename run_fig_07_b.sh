#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)*0.6" | bc`
num_threads=$(round ${num_threads} 0)
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

exe=build/examples/test_scalability

eps=0.001
num_sec=16

outdir=result/kl_res/scalability
mkdir -p ${outdir}

for Ns in 1000000 0100000 0200000 0400000 0600000 0800000; do
for rho in 2.0 2.5 3.0 3.5; do 
${exe} outdir=${outdir} num_sources=${Ns} num_targets=${Nq} num_sec=${num_sec} eps=${eps} rho=${rho} max_supernode_size=10000000 pcg_iters=${pcg_iters} count_loss=false 2>&1 | tee ${outdir}/log-Ns-${Ns}-rho-${rho}.txt
done
done
