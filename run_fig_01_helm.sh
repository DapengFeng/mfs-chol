#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)*0.6" | bc`
num_threads=$(round ${num_threads} 0)
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

exe_1=build/examples/test_helmholtz_waves

mesh=dat/island.off
query_mesh=dat/island-plane.off
solid_mesh=dat/island.off

filename=$(basename "${mesh}")
name="${filename%.*}"
echo "${name}"

rho=8.0
pcg_iters=40
eps=0.0001

for helm_k in 300; do #090.0 250.0 605.0; do

outdir=result/kl_res/helmholtz_waves_new/${name}/${helm_k}
mkdir -p ${outdir}

${exe_1} source_mesh=${mesh} target_mesh=${query_mesh} outdir=${outdir} num_sec=16 eps=${eps} rho=${rho} max_supernode_size=1000000 pcg_iters=${pcg_iters} helm_k=${helm_k} jacobi_precond=false solid_mesh=${solid_mesh} 2>&1 | tee ${outdir}/log-dirichlet-${name}-k-${helm_k}.txt

done
