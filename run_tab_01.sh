#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)*0.6" | bc`
num_threads=$(round ${num_threads} 0)
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

exe_1=build/examples/test_direct_svd

for source_num in 01 02 04 06 08; do
source_mesh=dat/helm_svd_source_${source_num}.off
boundary_mesh=${source_mesh}
target_mesh=dat/helm_svd_target.off

filename=$(basename "${source_mesh}")
name="${filename%.*}"
echo "${name}"
    
outdir=result/kl_res/helmholtz_direct_svd/${name}/
sec_num=16

helm_k=300.0
rho=8.0
pcg_iters=15
eps=0.0001
use_svd=true

mkdir -p ${outdir}

${exe_1} source_mesh=${source_mesh} target_mesh=${target_mesh} boundary_mesh=${boundary_mesh} outdir=${outdir} sec_num=${sec_num} rho=${rho} max_supernode_size=1000000 pcg_iters=${pcg_iters} num_sec=${sec_num} helm_k=${helm_k} eps=${eps} use_svd=${use_svd} 2>&1 | tee ${outdir}/log-${name}.txt
done
