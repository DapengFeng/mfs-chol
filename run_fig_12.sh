#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)*0.6" | bc`
num_threads=$(round ${num_threads} 0)
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

exe_1=build/examples/test_helmholtz_waves
exe_2=build/examples/test_helmholtz_waves_neumann

mesh=dat/square-highres.off
mesh=dat/circle-highres.off
query_mesh=dat/rect_query_plane.off

filename=$(basename "${mesh}")
name="${filename%.*}"
echo "${name}"

helm_k=300.0
rho=8.0

if (( ${1} == 0 )); then
    pcg_iters=20
    eps=0.001
    outdir=result/kl_res/dirichlet_neumann/${name}/D/
    mkdir -p ${outdir}
${exe_1} source_mesh=${mesh} target_mesh=${query_mesh} outdir=${outdir} eps=${eps} rho=${rho} max_supernode_size=1000000 pcg_iters=${pcg_iters} num_sec=16 helm_k=${helm_k} solid_mesh=
fi

if (( ${1} == 1 )); then
    pcg_iters=15
    eps=0.0024
    outdir=result/kl_res/dirichlet_neumann/${name}/N/
    mkdir -p ${outdir}	
${exe_2} source_mesh=${mesh} target_mesh=${query_mesh} outdir=${outdir} eps=${eps} rho=${rho} max_supernode_size=1000000 pcg_iters=${pcg_iters} num_sec=16 helm_k=${helm_k}
fi
