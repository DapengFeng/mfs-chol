#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)*0.6" | bc`
num_threads=$(round ${num_threads} 0)
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

exe=build/examples/test_scalability

eps=0.005
num_sec=16
Ns=13000
Nq=50000

outdir=result/kl_res/statistics
mkdir -p ${outdir}

for rho in 1.5 2.5 3.5 4.5 5.5 6.5 7.5; do 
${exe} outdir=${outdir} num_sources=${Ns} num_targets=${Nq} num_sec=${num_sec} eps=${eps} rho=${rho} max_supernode_size=10000000 pcg_iters=${pcg_iters} count_loss=true 2>&1 | tee ${outdir}/log-Ns-${Ns}-rho-${rho}.txt
done
